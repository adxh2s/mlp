from __future__ import annotations
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
  ou via des alias courants (svc, random_forest), afin de garantir la présence
  d’une méthode predict/decision_function en dernière étape.
"""

import importlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

# Encoders optionnels (Hashing/Target) si disponibles
try:
    from sklearn.feature_extraction import FeatureHasher
except Exception:  # noqa: BLE001
    FeatureHasher = None  # type: ignore[assignment]

try:
    # Présent dans scikit-learn récents
    from sklearn.preprocessing import TargetEncoder  # type: ignore[attr-defined]
    HAS_SK_TARGET = True
except Exception:  # noqa: BLE001
    TargetEncoder = None  # type: ignore[assignment]
    HAS_SK_TARGET = False

from src.preprocessing.reducers import ReducersFactory
from src.preprocessing.selectors import SelectorsFactory

# Noms d’étapes
STEP_PREPROCESS = "preprocess"
STEP_CT = "ct"
STEP_PRE_PCA_IMPUTER = "pre_pca_imputer"
STEP_FEAT_SEL = "feat_sel"
STEP_REDUCTION = "reduction"
STEP_ESTIMATOR = "estimator"

# Clés config
KEY_PARAMS = "params"
KEY_DISTS = "distributions"
KEY_TYPE = "type"
KEY_POLICY = "policy"


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
    def _wrap_list(v: Any) -> List[Any]:
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _build_numeric_scaler(scaler_name: str | None) -> Any:
        """Retourne un scaler numérique depuis une chaîne."""
        name = str(scaler_name or "auto")
        if name == "RobustScaler":
            return RobustScaler()
        if name == "MinMaxScaler":
            return MinMaxScaler()
        # auto / StandardScaler
        return StandardScaler()

    @staticmethod
    def _build_categorical_encoder(cfg: Dict[str, Any]) -> Any:
        """Retourne l’encodeur catégoriel selon la config locale."""
        enc = str(cfg.get("encoder", "OneHotEncoder"))
        if enc == "Hashing" and FeatureHasher is not None:
            n_features = int(cfg.get("n_features", 1024))
            return FeatureHasher(n_features=n_features, input_type="string")
        if enc == "TargetEncoder" and HAS_SK_TARGET:
            params = {k: v for k, v in cfg.items() if k != "encoder"}
            return TargetEncoder(**params)  # type: ignore[call-arg]
        # OneHot par défaut (sécurisé en prod)
        params: Dict[str, Any] = {}
        if "handle_unknown" in cfg:
            params["handle_unknown"] = cfg.get("handle_unknown", "ignore")
        if "min_frequency" in cfg:
            params["min_frequency"] = cfg["min_frequency"]
        # Note: pour sklearn < 1.2, remplacer sparse_output=False par sparse=False si besoin
        return OneHotEncoder(**params, sparse_output=False)

    @staticmethod
    def _selector_from_rule(rule: Dict[str, Any]) -> Union[List[str], Any]:
        """Construit un sélecteur de colonnes depuis une règle (noms/regex)."""
        if "include" in rule:
            return rule["include"]
        if "include_regex" in rule:
            pattern = re.compile(str(rule["include_regex"]))

            def regex_selector(X):  # type: ignore[no-untyped-def]
                return [c for c in X.columns if pattern.search(c)]

            return regex_selector
        # fallback
        return make_column_selector(dtype_include=object)

    @classmethod
    def _make_num_pipe(cls, local_num: Dict[str, Any] | None, fallback_num: Dict[str, Any] | None) -> Any:
        """Construit le sous-pipeline numérique (imputer + scaler) selon les overrides locaux/politiques globales."""
        local_num = local_num or {}
        fallback_num = fallback_num or {}
        steps: List[Tuple[str, Any]] = []
        imputer_flag = local_num.get("imputer") or fallback_num.get("imputer")
        if str(imputer_flag) == "simple":
            steps.append(("imputer", SimpleImputer(strategy="median")))
        scaler_name = local_num.get("scaler", fallback_num.get("scaler", "auto"))
        steps.append(("scaler", cls._build_numeric_scaler(scaler_name)))
        return Pipeline(steps) if steps else "passthrough"

    @classmethod
    def _make_cat_pipe(cls, local_cat: Dict[str, Any] | None, fallback_cat: Dict[str, Any] | None) -> Any:
        """Construit le sous-pipeline catégoriel (imputer + encodeur) selon les overrides locaux/politiques globales."""
        local_cat = local_cat or {}
        fallback_cat = fallback_cat or {}
        merged = {**fallback_cat, **local_cat}
        steps: List[Tuple[str, Any]] = []
        imputer_flag = merged.get("imputer")
        if str(imputer_flag) == "simple":
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        steps.append(("encoder", cls._build_categorical_encoder(merged)))
        return Pipeline(steps) if steps else "passthrough"

    @classmethod
    def _build_column_transformer(
        cls, spec: Dict[str, Any], global_policy: Dict[str, Any]
    ) -> Optional[ColumnTransformer]:
        """
        Construit un ColumnTransformer en première étape, avec imputation intégrée.
        - policy: "manual" (règles de colonnes) ou "auto" (sélection par dtype).
        - numeric: imputer: simple => SimpleImputer(median), scaler configurable.
        - categorical: imputer: simple => SimpleImputer(most_frequent), encoder configurable.
        """
        pre = (spec.get("steps") or {}).get(STEP_PREPROCESS) or {}
        ct_cfg = (pre.get("column_transformer") or {})
        if not ct_cfg or ct_cfg.get("enabled", True) is False:
            return None

        policy = ct_cfg.get(KEY_POLICY, "auto")
        transformers: List[Tuple[str, Any, Any]] = []

        if policy == "manual":
            for rule in ct_cfg.get("columns", []):
                if "numeric" in rule:
                    sel = cls._selector_from_rule(rule)
                    num_pipe = cls._make_num_pipe(rule.get("numeric"), global_policy.get("numeric"))
                    transformers.append(("num", num_pipe, sel))
                if "categorical" in rule:
                    sel = cls._selector_from_rule(rule)
                    cat_pipe = cls._make_cat_pipe(rule.get("categorical"), global_policy.get("categorical"))
                    transformers.append(("cat", cat_pipe, sel))
        else:
            # auto: dispatcher par dtype + overrides locaux
            local_num = ct_cfg.get("numeric") or {}
            local_cat = ct_cfg.get("categorical") or {}
            num_pipe = cls._make_num_pipe(local_num, global_policy.get("numeric"))
            cat_pipe = cls._make_cat_pipe(local_cat, global_policy.get("categorical"))
            transformers = [
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=["object", "category"])),
            ]

        # remainder='drop' pour ne garder que les colonnes traitées
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    @staticmethod
    def _flatten_grid(prefix: str, params: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Aplati un dictionnaire de paramètres -> param_grid sklearn (step__param)."""
        grid: Dict[str, List[Any]] = {}
        for k, v in (params or {}).items():
            grid[f"{prefix}__{k}"] = v if isinstance(v, list) else [v]
        return grid

    @staticmethod
    def _scipy_dist(name: str, low: float, high: float):
        """Construit une distribution scipy.stats depuis un spec déclaratif."""
        from scipy.stats import loguniform, uniform  # lazy import

        if name == "loguniform":
            return loguniform(low, high)
        if name == "uniform":
            return uniform(loc=low, scale=(high - low))
        msg = f"Unsupported dist: {name}"
        raise ValueError(msg)

    @classmethod
    def _flatten_distributions(cls, prefix: str, dists: Dict[str, Any]) -> Dict[str, Any]:
        """Aplati des distributions -> param_distributions sklearn (step__param: dist)."""
        out: Dict[str, Any] = {}
        for k, spec in (dists or {}).items():
            if not isinstance(spec, dict) or "dist" not in spec:
                continue
            dist = cls._scipy_dist(str(spec["dist"]), float(spec.get("low", 0.0)), float(spec.get("high", 1.0)))
            out[f"{prefix}__{k}"] = dist
        return out

    @staticmethod
    def _instantiate_estimator(est_cfg: Dict[str, Any]) -> Any:
        """
        Crée l'estimateur à partir de est_cfg['type'] (chemin fully-qualified ou alias).
        Garantit la présence de predict/decision_function pour la compatibilité GridSearchCV.
        """
        t = (est_cfg or {}).get("type")
        if not t:
            raise ValueError("Estimator type is required in spec.steps.estimator.type")
        # Fully-qualified import (ex: sklearn.svm.SVC)
        if "." in str(t):
            module, cls_name = str(t).rsplit(".", 1)
            klass = getattr(importlib.import_module(module), cls_name)
            model = klass()
        else:
            # Aliases minimaux
            alias = str(t).lower()
            if alias in {"svc", "svm"}:
                from sklearn.svm import SVC

                model = SVC()
            elif alias in {"rf", "random_forest", "randomforestclassifier"}:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier()
            else:
                raise ValueError(f"Unknown estimator alias/type: {t}")

        # sécurité: vérifier predict / decision_function
        if not hasattr(model, "fit") or not (hasattr(model, "predict") or hasattr(model, "decision_function")):
            raise TypeError(f"Estimator {type(model)} does not implement predict/decision_function")
        return model

    # -------------------------
    # Entrée principale
    # -------------------------
    @classmethod
    def build(
        cls, spec: Dict[str, Any], global_policy: Dict[str, Any]
    ) -> Tuple[Pipeline, Dict[str, List[Any]], Dict[str, Any]]:
        """
        Construit (pipeline, param_grid, param_distributions) depuis un spec.
        - Ordre garanti: ColumnTransformer (avec imputation) -> feature_selection
          -> pre_pca_imputer -> reduction -> estimator.
        - Les hyperparamètres sont adressés avec step__param selon sklearn.
        - Les distributions sont destinées à Randomized/HalvingRandom.
        """
        steps: List[Tuple[str, Any]] = []

        # ColumnTransformer (toujours en premier si présent)
        ct = cls._build_column_transformer(spec, global_policy)
        if ct is not None:
            steps.append((STEP_CT, ct))

        # Feature selection (optionnelle)
        feat_sel = (spec.get("steps") or {}).get(STEP_FEAT_SEL) or {}
        if feat_sel.get("enabled", False):
            sel = SelectorsFactory.from_spec(feat_sel)
            steps.append((STEP_FEAT_SEL, sel))

        # Sécurité contre NaN avant PCA + Réduction
        red = (spec.get("steps") or {}).get(STEP_REDUCTION) or {}
        if red.get(KEY_TYPE):
            steps.append((STEP_PRE_PCA_IMPUTER, SimpleImputer(strategy="median")))
            reducer = ReducersFactory.from_spec(red)
            steps.append((STEP_REDUCTION, reducer))

        # Estimateur
        est = (spec.get("steps") or {}).get(STEP_ESTIMATOR) or {}
        model = None
        if not spec.get("automl"):
            model = cls._instantiate_estimator(est)
        if model is not None:
            steps.append((STEP_ESTIMATOR, model))

        pipe = Pipeline(steps=[s for s in steps if s[1] is not None])

        # Grilles/distributions
        grid: Dict[str, List[Any]] = {}
        dists: Dict[str, Any] = {}

        pre = (spec.get("steps") or {}).get(STEP_PREPROCESS) or {}

        # ColumnTransformer sous-étapes (ex: ct__cat__encoder__min_frequency)
        ct_params = ((pre.get("column_transformer") or {}).get(KEY_PARAMS)) or {}
        grid.update(cls._flatten_grid(STEP_CT, ct_params))

        # feature selection
        fs_params = (feat_sel or {}).get(KEY_PARAMS) or {}
        grid.update(cls._flatten_grid(STEP_FEAT_SEL, fs_params))

        # reduction
        red_params = (red or {}).get(KEY_PARAMS) or {}
        grid.update(cls._flatten_grid(STEP_REDUCTION, red_params))

        # estimator
        est_params = (est or {}).get(KEY_PARAMS) or {}
        grid.update(cls._flatten_grid(STEP_ESTIMATOR, est_params))

        # estimator distributions
        est_dists = (est or {}).get(KEY_DISTS) or {}
        dists.update(cls._flatten_distributions(STEP_ESTIMATOR, est_dists))

        return pipe, grid, dists
