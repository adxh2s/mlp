from __future__ import annotations

"""
Fabrique de pipelines: construit sklearn.Pipeline, ColumnTransformer,
et normalise param_grid/param_distributions à partir d’une spécification YAML.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.feature_extraction import FeatureHasher

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
STEP_IMPUTER = "imputer"
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
    - ColumnTransformer auto/manuelle via selectors (noms/regex/dtypes).
    - Grilles/distributions compatibles Grid/Random/Halving.
    - Étapes optionnelles ignorées si enabled=false.
    """

    @staticmethod
    def _wrap_list(v: Any) -> List[Any]:
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _build_numeric_scaler(policy_num: Dict[str, Any]) -> Any:
        """Retourne le scaler numérique selon la politique globale."""
        scaler = str(policy_num.get("scaler", "auto"))
        if scaler == "RobustScaler":
            return RobustScaler()
        if scaler == "MinMaxScaler":
            return MinMaxScaler()
        # auto / StandardScaler
        return StandardScaler()

    @staticmethod
    def _build_categorical_encoder(cfg: Dict[str, Any]) -> Any:
        """Retourne l’encodeur catégoriel selon la config locale."""
        enc = str(cfg.get("encoder", "OneHotEncoder"))
        if enc == "Hashing":
            n_features = int(cfg.get("n_features", 1024))
            return FeatureHasher(n_features=n_features, input_type="string")
        if enc == "TargetEncoder":
            if HAS_SK_TARGET:
                params = {k: v for k, v in cfg.items() if k not in ("encoder",)}
                return TargetEncoder(**params)  # type: ignore[call-arg]
            # fallback OneHot si indisponible
            enc = "OneHotEncoder"

        # OneHot par défaut (sécurisé en prod)
        params: Dict[str, Any] = {}
        if "handle_unknown" in cfg:
            params["handle_unknown"] = cfg.get("handle_unknown", "ignore")
        if "min_frequency" in cfg:
            params["min_frequency"] = cfg["min_frequency"]
        return OneHotEncoder(**params)

    @staticmethod
    def _selector_from_rule(rule: Dict[str, Any]) -> Union[List[str], Any]:
        """Construit un sélecteur de colonnes depuis une règle (noms/regex)."""
        if "include" in rule:
            cols = rule["include"]
            return cols
        if "include_regex" in rule:
            pattern = re.compile(str(rule["include_regex"]))

            def regex_selector(X):  # type: ignore[no-untyped-def]
                return [c for c in X.columns if pattern.search(c)]

            return regex_selector
        return make_column_selector(dtype_include=object)

    @classmethod
    def _build_column_transformer(cls, spec: Dict[str, Any], global_policy: Dict[str, Any]) -> Optional[ColumnTransformer]:
        ct_cfg = (((spec.get("steps") or {}).get(STEP_PREPROCESS) or {}).get("column_transformer") or {})
        if not ct_cfg or ct_cfg.get("enabled", True) is False:
            return None

        policy = ct_cfg.get(KEY_POLICY, "auto")
        transformers = []

        if policy == "manual":
            for rule in ct_cfg.get("columns", []):
                if "categorical" in rule:
                    sel = cls._selector_from_rule(rule)
                    enc = cls._build_categorical_encoder(rule["categorical"])
                    transformers.append(("cat", enc, sel))
                if "numeric" in rule:
                    sel = cls._selector_from_rule(rule)
                    scaler = cls._build_numeric_scaler(rule["numeric"])
                    transformers.append(("num", scaler, sel))
        else:
            # auto: dispatcher par dtype
            num_scaler = cls._build_numeric_scaler((global_policy.get("numeric") or {}))
            cat_enc = cls._build_categorical_encoder((global_policy.get("categorical") or {}))
            transformers = [
                ("num", num_scaler, make_column_selector(dtype_include=np.number)),
                ("cat", cat_enc, make_column_selector(dtype_include=object)),
            ]

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @staticmethod
    def _simple_imputer(name: str = "simple") -> SimpleImputer:
        """Construit un imputer simple depuis un alias de stratégie."""
        strategy = "most_frequent" if name == "simple" else name
        return SimpleImputer(strategy=strategy)

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

    @classmethod
    def build(cls, spec: Dict[str, Any], global_policy: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, List[Any]], Dict[str, Any]]:
        """
        Construit (pipeline, param_grid, param_distributions) depuis un spec.

        - Les hyperparamètres sont adressés avec step__param selon sklearn.
        - Les distributions sont destinées à Randomized/HalvingRandom.
        """
        steps: List[Tuple[str, Any]] = []

        # preprocess
        pre = (spec.get("steps") or {}).get(STEP_PREPROCESS) or {}
        if pre.get("enabled", True):
            if "imputer" in pre:
                steps.append((STEP_IMPUTER, cls._simple_imputer(str(pre["imputer"]))))
            ct = cls._build_column_transformer(spec, global_policy)
            if ct is not None:
                steps.append((STEP_CT, ct))

        # feature selection
        feat_sel = (spec.get("steps") or {}).get(STEP_FEAT_SEL) or {}
        if feat_sel.get("enabled", False):
            sel = SelectorsFactory.from_spec(feat_sel)
            steps.append((STEP_FEAT_SEL, sel))

        # reduction
        red = (spec.get("steps") or {}).get(STEP_REDUCTION) or {}
        if red.get(KEY_TYPE):
            reducer = ReducersFactory.from_spec(red)
            steps.append((STEP_REDUCTION, reducer))

        # estimator
        est = (spec.get("steps") or {}).get(STEP_ESTIMATOR) or {}
        if spec.get("automl"):
            model = ("passthrough", None)  # géré par l’évaluateur
        else:
            model = SelectorsFactory.instantiate_estimator(est)
        steps.append((STEP_ESTIMATOR, model))

        pipe = Pipeline(steps=[s for s in steps if s[1] is not None])

        # Grilles/distributions
        grid: Dict[str, List[Any]] = {}
        dists: Dict[str, Any] = {}

        # ColumnTransformer sous-étapes (ex: ct__cat__min_frequency)
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
