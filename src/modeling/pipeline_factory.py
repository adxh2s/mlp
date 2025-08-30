from __future__ import annotations
from typing import Any, Dict, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from ..preprocessing.selectors import SelectorsFactory
from ..preprocessing.reducers import ReducersFactory

class PipelineFactory:
    STEP_PREPROCESS = "preprocess"
    STEP_FEAT_SEL = "feat_sel"
    STEP_REDUCTION = "reduction"
    STEP_ESTIMATOR = "estimator"
    KEY_PARAMS = "params"
    PASSTHROUGH = "passthrough"
    IMPUTER_SIMPLE = "simple"
    SCALER_STANDARD = "standard"
    ESTIMATOR_SVC = "svc"
    ESTIMATOR_RF = "random_forest"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _preprocess(self, cfg: Dict[str, Any]):
        steps = []
        if cfg.get("imputer") == self.IMPUTER_SIMPLE:
            steps.append(("imputer", SimpleImputer(strategy="median")))
        if cfg.get("scaler") == self.SCALER_STANDARD:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps) if steps else self.PASSTHROUGH

    def _estimator(self, cfg: Dict[str, Any]):
        etype = cfg.get("type")
        if etype == self.ESTIMATOR_SVC:
            return SVC(random_state=self.random_state)
        if etype == self.ESTIMATOR_RF:
            return RandomForestClassifier(random_state=self.random_state)
        raise ValueError(f"Unknown estimator {etype}")

    def build(self, spec: Dict[str, Any]) -> Tuple[Any, Dict[str, List[Any]]]:
        if "automl" in spec:
            try:
                import autosklearn.classification as askl_cls
            except Exception:
                raise RuntimeError("auto-sklearn not installed")
            a = spec["automl"]
            model = askl_cls.AutoSklearnClassifier(
                time_left_for_this_task=a.get("time_left_for_this_task", 300),
                per_run_time_limit=a.get("per_run_time_limit", 60),
                seed=self.random_state
            )
            return model, {}

        steps_cfg = spec.get("steps") or {}
        preprocess = self._preprocess(steps_cfg.get(self.STEP_PREPROCESS) or {})
        selector = SelectorsFactory.make_selector(steps_cfg.get("feature_selection"))
        reducer = ReducersFactory.make_reducer(steps_cfg.get(self.STEP_REDUCTION), self.random_state)
        estimator = self._estimator(steps_cfg.get(self.STEP_ESTIMATOR) or {})

        steps_list = []
        if preprocess != self.PASSTHROUGH:
            steps_list.append((self.STEP_PREPROCESS, preprocess))
        if selector != self.PASSTHROUGH:
            steps_list.append((self.STEP_FEAT_SEL, selector))
        if reducer != self.PASSTHROUGH:
            steps_list.append((self.STEP_REDUCTION, reducer))
        steps_list.append((self.STEP_ESTIMATOR, estimator))
        pipe = Pipeline(steps_list)

        grid: Dict[str, List[Any]] = {}
        red_params = (steps_cfg.get(self.STEP_REDUCTION) or {}).get(self.KEY_PARAMS) or {}
        for k, v in red_params.items():
            grid[f"{self.STEP_REDUCTION}__{k}"] = v
        est_params = (steps_cfg.get(self.STEP_ESTIMATOR) or {}).get(self.KEY_PARAMS) or {}
        for k, v in est_params.items():
            grid[f"{self.STEP_ESTIMATOR}__{k}"] = v
        return pipe, grid
