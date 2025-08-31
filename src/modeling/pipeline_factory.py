from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..preprocessing.reducers import ReducersFactory
from ..preprocessing.selectors import SelectorsFactory


class PipelineFactory:
    """
    Build scikit-learn Pipelines and parameter grids from a simple spec.

    Features:
    - Optional AutoML backends (AutoSklearn/TPOT) when requested in spec["automl"].
    - Preprocessing steps (imputer, scaler), optional selection/reduction stages.
    - Normalized GridSearchCV param grid: scalars are wrapped into 1-element lists.
    """

    # Step names
    STEP_PREPROCESS = "preprocess"
    STEP_FEAT_SEL = "feat_sel"
    STEP_REDUCTION = "reduction"
    STEP_ESTIMATOR = "estimator"

    # Config keys
    KEY_PARAMS = "params"
    KEY_AUTOML = "automl"
    KEY_TYPE = "type"

    # Passthrough marker
    PASSTHROUGH = "passthrough"

    # Preprocess constants
    IMPUTER_SIMPLE = "simple"
    SCALER_STANDARD = "standard"

    # Estimator types
    ESTIMATOR_SVC = "svc"
    ESTIMATOR_RF = "random_forest"

    # AutoML types
    AUTOML_AUTOSKLEARN = "autosklearn"
    AUTOML_TPOT = "tpot"

    # Auto-sklearn defaults
    AKL_TIME_LEFT = "time_left_for_this_task"
    AKL_PER_RUN_LIMIT = "per_run_time_limit"
    AKL_DEFAULT_TIME_LEFT = 300
    AKL_DEFAULT_PER_RUN = 60

    # TPOT defaults
    TPOT_GENERATIONS = "generations"
    TPOT_POP_SIZE = "population_size"
    TPOT_SCORING = "scoring"
    TPOT_CV = "cv"
    TPOT_DEFAULT_GENERATIONS = 5
    TPOT_DEFAULT_POP_SIZE = 20
    TPOT_DEFAULT_SCORING = "f1"
    TPOT_DEFAULT_CV = 5
    TPOT_VERBOSITY = 2
    TPOT_N_JOBS = -1

    # Error messages
    ERR_UNKNOWN_ESTIMATOR = "Unknown estimator {}"
    ERR_UNKNOWN_AUTOML = "Unknown automl type: {}"
    ERR_AUTOSKLEARN_MISSING = "auto-sklearn not installed"
    ERR_TPOT_MISSING = "TPOT not installed"

    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the factory.

        Args:
            random_state: Seed to pass to estimators/reducers when applicable.
        """
        self.random_state = random_state

    def _preprocess(self, cfg: Dict[str, Any]) -> Pipeline | str:
        """
        Build the preprocessing pipeline from config.

        Steps supported:
        - SimpleImputer(strategy="median") when imputer == "simple"
        - StandardScaler() when scaler == "standard"

        Returns:
            A Pipeline with the configured steps or PASSTHROUGH if none.
        """
        steps: list[tuple[str, Any]] = []
        if cfg.get("imputer") == self.IMPUTER_SIMPLE:
            steps.append(("imputer", SimpleImputer(strategy="median")))
        if cfg.get("scaler") == self.SCALER_STANDARD:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps) if steps else self.PASSTHROUGH

    def _estimator(self, cfg: Dict[str, Any]) -> Any:
        """
        Instantiate the estimator selected in config.

        Supported:
        - "svc" -> sklearn.svm.SVC
        - "random_forest" -> sklearn.ensemble.RandomForestClassifier
        """
        etype = cfg.get(self.KEY_TYPE)
        if etype == self.ESTIMATOR_SVC:
            return SVC(random_state=self.random_state)
        if etype == self.ESTIMATOR_RF:
            return RandomForestClassifier(random_state=self.random_state)
        raise ValueError(self.ERR_UNKNOWN_ESTIMATOR.format(etype))

    @staticmethod
    def _to_seq(value: Any) -> List[Any] | np.ndarray:
        """
        Normalize a hyperparameter value to a sequence acceptable by GridSearchCV.

        - numpy.ndarray (assumed 1D): returned as is.
        - list/tuple: cast to list.
        - scalar: wrapped into a single-element list.

        Returns:
            A sequence (list or ndarray) suitable for param_grid.
        """
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _build_automl(self, automl_cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, List[Any]]]:
        """
        Build an AutoML model if requested by config.

        Supports:
        - type == "autosklearn": returns AutoSklearnClassifier
        - type == "tpot": returns TPOTClassifier

        Returns:
            (model, param_grid) with an empty grid for AutoML models.
        """
        a_type = (automl_cfg.get(self.KEY_TYPE) or "").lower()

        if a_type == self.AUTOML_AUTOSKLEARN:
            try:
                import autosklearn.classification as askl_cls
            except Exception:
                raise RuntimeError(self.ERR_AUTOSKLEARN_MISSING)
            model = askl_cls.AutoSklearnClassifier(
                time_left_for_this_task=automl_cfg.get(self.AKL_TIME_LEFT, self.AKL_DEFAULT_TIME_LEFT),
                per_run_time_limit=automl_cfg.get(self.AKL_PER_RUN_LIMIT, self.AKL_DEFAULT_PER_RUN),
                seed=self.random_state,
            )
            return model, {}

        if a_type == self.AUTOML_TPOT:
            try:
                from tpot import TPOTClassifier
            except Exception:
                raise RuntimeError(self.ERR_TPOT_MISSING)
            model = TPOTClassifier(
                generations=automl_cfg.get(self.TPOT_GENERATIONS, self.TPOT_DEFAULT_GENERATIONS),
                population_size=automl_cfg.get(self.TPOT_POP_SIZE, self.TPOT_DEFAULT_POP_SIZE),
                scoring=automl_cfg.get(self.TPOT_SCORING, self.TPOT_DEFAULT_SCORING),
                cv=automl_cfg.get(self.TPOT_CV, self.TPOT_DEFAULT_CV),
                random_state=self.random_state,
                verbosity=self.TPOT_VERBOSITY,
                n_jobs=self.TPOT_N_JOBS,
            )
            return model, {}

        raise ValueError(self.ERR_UNKNOWN_AUTOML.format(a_type))

    def build(self, spec: Dict[str, Any]) -> Tuple[Any, Dict[str, List[Any]]]:
        """
        Build either an AutoML model or a Pipeline + param grid from spec.

        Spec format (abridged):
        - automl: {type: "tpot"|"autosklearn", ...}  # optional
        - steps:
            preprocess: {imputer: "simple", scaler: "standard"}
            feature_selection: {...}
            reduction: {type: ..., params: {...}}
            estimator: {type: "svc"|"random_forest", params: {...}}

        Returns:
            A tuple (model_or_pipeline, param_grid).
        """
        automl_cfg = spec.get(self.KEY_AUTOML)
        if automl_cfg:
            return self._build_automl(automl_cfg)

        steps_cfg = spec.get("steps") or {}
        preprocess = self._preprocess(steps_cfg.get(self.STEP_PREPROCESS) or {})
        selector = SelectorsFactory.make_selector(steps_cfg.get("feature_selection"))
        reducer = ReducersFactory.make_reducer(steps_cfg.get(self.STEP_REDUCTION), self.random_state)
        estimator = self._estimator(steps_cfg.get(self.STEP_ESTIMATOR) or {})

        steps_list: list[tuple[str, Any]] = []
        if preprocess != self.PASSTHROUGH:
            steps_list.append((self.STEP_PREPROCESS, preprocess))
        if selector != self.PASSTHROUGH:
            steps_list.append((self.STEP_FEAT_SEL, selector))
        if reducer != self.PASSTHROUGH:
            steps_list.append((self.STEP_REDUCTION, reducer))
        steps_list.append((self.STEP_ESTIMATOR, estimator))
        pipe = Pipeline(steps_list)

        grid: Dict[str, List[Any]] = {}

        # Reduction params -> normalize to sequences
        red_params = (steps_cfg.get(self.STEP_REDUCTION) or {}).get(self.KEY_PARAMS) or {}
        for k, v in red_params.items():
            grid[f"{self.STEP_REDUCTION}__{k}"] = self._to_seq(v)

        # Estimator params -> normalize to sequences
        est_params = (steps_cfg.get(self.STEP_ESTIMATOR) or {}).get(self.KEY_PARAMS) or {}
        for k, v in est_params.items():
            grid[f"{self.STEP_ESTIMATOR}__{k}"] = self._to_seq(v)

        return pipe, grid
