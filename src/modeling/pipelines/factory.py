from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar, cast, overload

import numpy as np
import pandas as pd
from sklearn import feature_extraction as _sk_fe
from sklearn import preprocessing as _sk_pre
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.preprocessing.reducers import ReducersFactory
from src.preprocessing.selectors import SelectorsFactory

"""
Fabrique de pipelines: construit sklearn.Pipeline, ColumnTransformer,
et normalise param_grid/param_distributions à partir d’une spécification YAML.

Principes clés:
- Le ColumnTransformer est la première étape et contient l’imputation/encodage,
  afin d’éviter d’envoyer des NaN à la PCA/aux estimateurs et de préserver
  les sélections de colonnes basées sur le dtype.
- Un imputer de sécurité "pre_pca_imputer" est inséré juste avant la réduction
  pour neutraliser tout NaN résiduel (au cas improbable d’une colonne non capturée).
- L’estimateur est instancié depuis un type fully-qualified (ex: sklearn.svm.SVC)
  ou via des aliases courants (svc, random_forest), afin de garantir la présence
  d’une méthode predict/decision_function en dernière étape.
"""

# =========================
# Constantes de chaînes
# =========================

# Étapes de pipeline
STEP_PREPROCESS = "preprocess"
STEP_CT = "ct"
STEP_PRE_PCA_IMPUTER = "pre_pca_imputer"
STEP_FEAT_SEL = "feat_sel"
STEP_REDUCTION = "reduction"
STEP_ESTIMATOR = "estimator"

# Clés de config génériques
STEPS_KEY = "steps"
KEY_PARAMS = "params"
KEY_DISTS = "distributions"
KEY_TYPE = "type"
KEY_POLICY = "policy"
KEY_ENABLED = "enabled"

# ColumnTransformer et sous-clés
KEY_CT_NAME = "column_transformer"
KEY_COLUMNS = "columns"
KEY_NUMERIC = "numeric"
KEY_CATEGORICAL = "categorical"

# Prétraitement catégoriel
ENCODER_KEY = "encoder"
ENCODER_HASHING = "Hashing"
ENCODER_TARGET = "TargetEncoder"
ENCODER_ONEHOT = "OneHotEncoder"
HANDLE_UNKNOWN_KEY = "handle_unknown"
MIN_FREQUENCY_KEY = "min_frequency"

# Imputation / scaling
IMPUTER_KEY = "imputer"
IMPUTER_SIMPLE = "simple"
SCALER_KEY = "scaler"
SCALER_AUTO = "auto"
SCALER_ROBUST = "RobustScaler"
SCALER_MINMAX = "MinMaxScaler"
SCALER_STANDARD = "StandardScaler"

# Sélecteurs de dtype
DTYPE_CATEGORICAL = ["object", "category"]
DTYPE_NUMERIC = np.number  # type: ignore[reportUnknownVariableType]

# Politiques CT
POLICY_AUTO = "auto"
POLICY_MANUAL = "manual"

# Distributions
DIST_LOGUNIFORM = "loguniform"
DIST_UNIFORM = "uniform"

# Aliases d’estimateurs
ALIAS_SVC = {"svc", "svm"}
ALIAS_RF = {"rf", "random_forest", "randomforestclassifier"}

# Messages d’erreur
MSG_UNSUPPORTED_DIST = "Unsupported dist: {name}"
MSG_REQ_ESTIMATOR_TYPE = "Estimator type is required in spec.steps.estimator.type"
MSG_UNKNOWN_ALIAS = "Unknown estimator alias/type: {t}"
MSG_NO_PREDICT = "Estimator {cls} does not implement predict/decision_function"

# Détection optionnelle des encodeurs (sans try/except d'import, pour éviter E402)
FeatureHasher = getattr(_sk_fe, "FeatureHasher", None)
TargetEncoder = getattr(_sk_pre, "TargetEncoder", None)
has_sk_target = TargetEncoder is not None

# Alias de type pour un sélecteur de colonnes
ColumnSelector = Callable[[pd.DataFrame], list[str]]

# =========================
# Helpers de typage sûrs
# =========================

T = TypeVar("T")


def wrap_list(v: T | list[T]) -> list[T]:
    """Renvoie v si déjà liste, sinon [v], en préservant le type élémentaire T."""
    if isinstance(v, list):
        return v
    return [cast(T, v)]


def wrap_list_any(v: Any) -> list[Any]:
    """Version tolérante pour valeurs Any (YAML/dicts hétérogènes) afin d'éviter Unknown."""
    return v if isinstance(v, list) else [v]


@overload
def as_str_list(items: None) -> list[str]: ...
@overload
def as_str_list(items: str | bytes) -> list[str]: ...
@overload
def as_str_list(items: Iterable[Any]) -> list[str]: ...
@overload
def as_str_list(items: Any) -> list[str]: ...


def as_str_list(items: Any) -> list[str]:
    """Convertit en list[str] en couvrant None/str/bytes/Iterable et en évitant les Unknown."""
    if items is None:
        return []
    if isinstance(items, (str, bytes)):
        return [str(items)]
    if isinstance(items, Iterable) and not isinstance(items, (str, bytes)):
        return [str(e) for e in cast(Iterable[object], items)]
    return [str(items)]


def as_mapping(obj: Mapping[str, Any] | None) -> dict[str, Any]:
    """Retourne un dict[str, Any] garanti à partir d'un Mapping optionnel, sans dict(obj)."""
    if obj is None:
        return {}
    return {str(k): v for (k, v) in obj.items()}


class PipelineFactory:
    """
    Construit un sklearn.Pipeline + grilles d’hyperparamètres à partir d’un spec.
    - ColumnTransformer auto/manuelle via selectors (noms/regex/dtypes), avec imputation intégrée.
    - Grilles/distributions compatibles Grid/Random/Halving.
    - Étapes optionnelles ignorées si enabled=false.
    """

    # -------------------------
    # Outils internes
    # -------------------------

    @staticmethod
    def _build_numeric_scaler(scaler_name: str | None) -> Any:
        """Retourne un scaler numérique depuis une chaîne."""
        name = str(scaler_name or SCALER_AUTO)
        if name == SCALER_ROBUST:
            return RobustScaler()
        if name == SCALER_MINMAX:
            return MinMaxScaler()
        return StandardScaler()

    @staticmethod
    def _build_categorical_encoder(cfg: Mapping[str, Any]) -> Any:
        """Retourne l’encodeur catégoriel selon la config locale."""
        enc = str(cfg.get(ENCODER_KEY, ENCODER_ONEHOT))
        if enc == ENCODER_HASHING and FeatureHasher is not None:
            n_features = int(cfg.get("n_features", 1024))
            return FeatureHasher(n_features=n_features, input_type="string")
        if enc == ENCODER_TARGET and has_sk_target and TargetEncoder is not None:
            params = {k: v for k, v in cfg.items() if k != ENCODER_KEY}
            return TargetEncoder(**params)  # type: ignore[misc]
        params: dict[str, Any] = {}
        if HANDLE_UNKNOWN_KEY in cfg:
            params[HANDLE_UNKNOWN_KEY] = cfg.get(HANDLE_UNKNOWN_KEY, "ignore")
        if MIN_FREQUENCY_KEY in cfg:
            params[MIN_FREQUENCY_KEY] = cfg[MIN_FREQUENCY_KEY]
        return OneHotEncoder(**params, sparse_output=False)

    @staticmethod
    def _selector_from_rule(rule: Mapping[str, Any]) -> ColumnSelector:
        """Construit un sélecteur de colonnes depuis une règle, toujours un callable ColumnSelector."""
        if "include" in rule:
            include_cols: list[str] = as_str_list(rule.get("include"))

            def include_selector(x: pd.DataFrame) -> list[str]:
                return include_cols

            return include_selector

        if "include_regex" in rule:
            pattern = re.compile(str(rule.get("include_regex", "")))

            def regex_selector(x: pd.DataFrame) -> list[str]:
                cols: list[str] = [str(c) for c in x.columns]
                return [c for c in cols if pattern.search(c)]

            return regex_selector

        def cat_selector(x: pd.DataFrame) -> list[str]:
            sel = make_column_selector(dtype_include=cast(Any, DTYPE_CATEGORICAL))
            cols_any = sel(x)
            cols: list[str] = [str(c) for c in cols_any]
            return cols

        return cat_selector

    @classmethod
    def _make_num_pipe(
        cls, local_num: Mapping[str, Any] | None, fallback_num: Mapping[str, Any] | None
    ) -> Any:
        """Construit le sous-pipeline numérique (imputer + scaler) selon overrides locaux/politiques globales."""
        local = as_mapping(local_num)
        fallback = as_mapping(fallback_num)
        steps: list[tuple[str, Any]] = []
        imputer_flag = local.get(IMPUTER_KEY, fallback.get(IMPUTER_KEY))
        if str(imputer_flag) == IMPUTER_SIMPLE:
            steps.append(("imputer", SimpleImputer(strategy="median")))
        scaler_name = cast(str | None, local.get(SCALER_KEY, fallback.get(SCALER_KEY, SCALER_AUTO)))
        steps.append(("scaler", cls._build_numeric_scaler(scaler_name)))
        return Pipeline(steps) if steps else "passthrough"

    @classmethod
    def _make_cat_pipe(
        cls, local_cat: Mapping[str, Any] | None, fallback_cat: Mapping[str, Any] | None
    ) -> Any:
        """Construit le sous-pipeline catégoriel (imputer + encodeur) selon overrides locaux/politiques globales."""
        merged: dict[str, Any] = {**as_mapping(fallback_cat), **as_mapping(local_cat)}
        steps: list[tuple[str, Any]] = []
        imputer_flag = merged.get(IMPUTER_KEY)
        if str(imputer_flag) == IMPUTER_SIMPLE:
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        steps.append(("encoder", cls._build_categorical_encoder(merged)))
        return Pipeline(steps) if steps else "passthrough"

    @classmethod
    def _build_column_transformer(
        cls, spec: Mapping[str, Any], global_policy: Mapping[str, Any]
    ) -> ColumnTransformer | None:
        """
        Construit un ColumnTransformer en première étape, avec imputation intégrée.
        - policy: "manual" (règles de colonnes) ou "auto" (sélection par dtype).
        - numeric: imputer: simple => SimpleImputer(median), scaler configurable.
        - categorical: imputer: simple => SimpleImputer(most_frequent), encoder configurable.
        """
        steps_cfg = as_mapping(cast(Mapping[str, Any] | None, spec.get(STEPS_KEY)))
        pre = as_mapping(cast(Mapping[str, Any] | None, steps_cfg.get(STEP_PREPROCESS)))
        ct_cfg = as_mapping(cast(Mapping[str, Any] | None, pre.get(KEY_CT_NAME)))
        if not ct_cfg or ct_cfg.get(KEY_ENABLED, True) is False:
            return None

        policy = ct_cfg.get(KEY_POLICY, POLICY_AUTO)
        transformers: list[tuple[str, Any, Any]] = []

        if policy == POLICY_MANUAL:
            for rule in cast(Sequence[Mapping[str, Any]], ct_cfg.get(KEY_COLUMNS, [])):
                if KEY_NUMERIC in rule:
                    sel = cls._selector_from_rule(rule)
                    num_pipe = cls._make_num_pipe(
                        cast(Mapping[str, Any] | None, rule.get(KEY_NUMERIC)),
                        cast(Mapping[str, Any] | None, global_policy.get(KEY_NUMERIC)),
                    )
                    transformers.append(("num", num_pipe, sel))
                if KEY_CATEGORICAL in rule:
                    sel = cls._selector_from_rule(rule)
                    cat_pipe = cls._make_cat_pipe(
                        cast(Mapping[str, Any] | None, rule.get(KEY_CATEGORICAL)),
                        cast(Mapping[str, Any] | None, global_policy.get(KEY_CATEGORICAL)),
                    )
                    transformers.append(("cat", cat_pipe, sel))
        else:
            local_num = cast(Mapping[str, Any] | None, ct_cfg.get(KEY_NUMERIC))
            local_cat = cast(Mapping[str, Any] | None, ct_cfg.get(KEY_CATEGORICAL))
            num_pipe = cls._make_num_pipe(
                local_num, cast(Mapping[str, Any] | None, global_policy.get(KEY_NUMERIC))
            )
            cat_pipe = cls._make_cat_pipe(
                local_cat, cast(Mapping[str, Any] | None, global_policy.get(KEY_CATEGORICAL))
            )
            num_selector = make_column_selector(dtype_include=cast(Any, DTYPE_NUMERIC))
            cat_selector = make_column_selector(dtype_include=cast(Any, DTYPE_CATEGORICAL))
            transformers = [
                ("num", num_pipe, num_selector),
                ("cat", cat_pipe, cat_selector),
            ]

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _flatten_grid(prefix: str, params: Mapping[str, Any]) -> dict[str, list[Any]]:
        """Aplati un dictionnaire de paramètres -> param_grid sklearn (step__param)."""
        p = as_mapping(cast(Mapping[str, Any] | None, params))
        grid: dict[str, list[Any]] = {}
        for k, v in p.items():
            grid[f"{prefix}__{k}"] = wrap_list_any(v)
        return grid

    @staticmethod
    def _scipy_dist(name: str, low: float, high: float) -> Any:
        """Construit une distribution scipy.stats depuis un spec déclaratif."""
        from scipy import stats as _stats  # type: ignore[reportMissingTypeStubs]  # noqa: PLC0415

        if name == DIST_LOGUNIFORM:
            return _stats.loguniform(low, high)
        if name == DIST_UNIFORM:
            return _stats.uniform(loc=low, scale=(high - low))
        msg = MSG_UNSUPPORTED_DIST.format(name=name)
        raise ValueError(msg)

    @classmethod
    def _flatten_distributions(cls, prefix: str, dists: Mapping[str, Any]) -> dict[str, Any]:
        """Aplati des distributions -> param_distributions sklearn (step__param: dist)."""
        d = as_mapping(cast(Mapping[str, Any] | None, dists))
        out: dict[str, Any] = {}
        for k, spec in d.items():
            sd = as_mapping(cast(Mapping[str, Any] | None, spec))
            if "dist" not in sd:
                continue
            dist = cls._scipy_dist(
                str(sd["dist"]), float(sd.get("low", 0.0)), float(sd.get("high", 1.0))
            )
            out[f"{prefix}__{k}"] = dist
        return out

    @staticmethod
    def _instantiate_estimator(est_cfg: Mapping[str, Any]) -> Any:
        """
        Crée l'estimateur à partir de est_cfg['type'] (chemin fully-qualified ou alias).
        Garantit la présence de predict/decision_function pour la compatibilité GridSearchCV.
        """
        t = est_cfg.get(KEY_TYPE)
        if not t:
            raise ValueError(MSG_REQ_ESTIMATOR_TYPE)

        if "." in str(t):
            module, cls_name = str(t).rsplit(".", 1)
            klass = getattr(importlib.import_module(module), cls_name)
            model = klass()
        else:
            alias = str(t).lower()
            if alias in ALIAS_SVC:
                from sklearn.svm import SVC  # noqa: PLC0415

                model = SVC()
            elif alias in ALIAS_RF:
                from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

                model = RandomForestClassifier()
            else:
                raise ValueError(MSG_UNKNOWN_ALIAS.format(t=t))

        if not hasattr(model, "fit") or not (
            hasattr(model, "predict") or hasattr(model, "decision_function")
        ):
            raise TypeError(MSG_NO_PREDICT.format(cls=type(model).__name__))
        return model

    @classmethod
    def build(
        cls, spec: Mapping[str, Any], global_policy: Mapping[str, Any]
    ) -> tuple[Pipeline, dict[str, list[Any]], dict[str, Any]]:
        """
        Construit (pipeline, param_grid, param_distributions) depuis un spec.
        - Ordre garanti: ColumnTransformer (avec imputation) -> feature_selection
          -> pre_pca_imputer -> reduction -> estimator.
        - Les hyperparamètres sont adressés avec step__param selon sklearn.
        - Les distributions sont destinées à Randomized/HalvingRandom.
        """
        steps_cfg = as_mapping(cast(Mapping[str, Any] | None, spec.get(STEPS_KEY)))

        steps: list[tuple[str, Any]] = []

        # ColumnTransformer (toujours en premier si présent)
        ct = cls._build_column_transformer(spec, global_policy)
        if ct is not None:
            steps.append((STEP_CT, ct))

        # Feature selection (optionnelle)
        feat_sel = as_mapping(cast(Mapping[str, Any] | None, steps_cfg.get(STEP_FEAT_SEL)))
        if feat_sel.get(KEY_ENABLED, False):
            sel = SelectorsFactory.from_spec(feat_sel)
            steps.append((STEP_FEAT_SEL, sel))

        # Sécurité contre NaN avant PCA + Réduction
        red = as_mapping(cast(Mapping[str, Any] | None, steps_cfg.get(STEP_REDUCTION)))
        if red.get(KEY_TYPE):
            steps.append((STEP_PRE_PCA_IMPUTER, SimpleImputer(strategy="median")))
            reducer = ReducersFactory.from_spec(red)
            steps.append((STEP_REDUCTION, reducer))

        # Estimateur
        est = as_mapping(cast(Mapping[str, Any] | None, steps_cfg.get(STEP_ESTIMATOR)))
        model = None
        if not as_mapping(cast(Mapping[str, Any] | None, spec.get("automl"))):
            model = cls._instantiate_estimator(est)
            if model is not None:
                steps.append((STEP_ESTIMATOR, model))

        pipe = Pipeline(steps=[s for s in steps if s[1] is not None])

        # Grilles/distributions
        grid: dict[str, list[Any]] = {}
        dists: dict[str, Any] = {}

        pre = as_mapping(cast(Mapping[str, Any] | None, steps_cfg.get(STEP_PREPROCESS)))
        ct_cfg = as_mapping(cast(Mapping[str, Any] | None, pre.get(KEY_CT_NAME)))

        # ColumnTransformer sous-étapes
        ct_params = as_mapping(cast(Mapping[str, Any] | None, ct_cfg.get(KEY_PARAMS)))
        grid.update(cls._flatten_grid(STEP_CT, ct_params))

        # feature selection
        fs_params = as_mapping(cast(Mapping[str, Any] | None, feat_sel.get(KEY_PARAMS)))
        grid.update(cls._flatten_grid(STEP_FEAT_SEL, fs_params))

        # reduction
        red_params = as_mapping(cast(Mapping[str, Any] | None, red.get(KEY_PARAMS)))
        grid.update(cls._flatten_grid(STEP_REDUCTION, red_params))

        # estimator
        est_params = as_mapping(cast(Mapping[str, Any] | None, est.get(KEY_PARAMS)))
        grid.update(cls._flatten_grid(STEP_ESTIMATOR, est_params))

        # estimator distributions
        est_dists = as_mapping(cast(Mapping[str, Any] | None, est.get(KEY_DISTS)))
        dists.update(cls._flatten_distributions(STEP_ESTIMATOR, est_dists))

        return pipe, grid, dists
