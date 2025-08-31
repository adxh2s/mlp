from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .pipeline_factory import PipelineFactory
from src.instrumentation.logger_mixin import LoggerMixin, SupportsGetLogger


class PipelineEvaluator(LoggerMixin):
    """Evaluate pipelines via GridSearchCV or TPOT AutoML, with optional MLflow logging."""

    DEFAULT_REFIT = "f1"
    FILE_PREFIX = "cv_"
    FILE_EXT = ".csv"

    def __init__(
        self,
        out_dir: str,
        random_state: int = 42,
        mlflow_enabled: bool = False,
        experiment: str = "mlp-experiments",
        logger_manager: SupportsGetLogger | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            out_dir: Output directory for artifacts.
            random_state: Seed for CV/reproducibility.
            mlflow_enabled: Enable MLflow tracking if True.
            experiment: MLflow experiment name.
            logger_manager: Optional LoggerManager/StructlogLoggerManager to initialize self.log.
        """
        self.out_dir = out_dir
        self.random_state = random_state
        self.mlflow_enabled = mlflow_enabled
        self.experiment = experiment

        # Initialize logger consistently
        if logger_manager is not None:
            self._init_logger(logger_manager)  # provides self.log (LoggerMixin)
        else:
            self.log = logging.getLogger("mlp.evaluator")

        os.makedirs(out_dir, exist_ok=True)

        if self.mlflow_enabled:
            uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", experiment))

    def _log_mlflow_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            mlflow.log_param(k, str(v))

    def _tpot_export(self, tpot_obj, export: bool, export_path: str) -> Optional[str]:
        try:
            if export:
                # export_path is relative to self.out_dir
                dest = os.path.join(self.out_dir, export_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tpot_obj.export(dest)
                return dest
        except Exception:
            # Silent export failure; continue
            return None
        return None

    def _evaluate_tpot(self, X, y, name: str, tcfg: Dict[str, Any]) -> Dict[str, Any]:
        """Run TPOT AutoML and return results dictionary."""
        from tpot import TPOTClassifier  # import here to keep dependency optional

        # Extract TPOT settings with sensible defaults
        generations = int(tcfg.get("generations", 5))
        population_size = int(tcfg.get("population_size", 50))
        max_time_mins = tcfg.get("max_time_mins")  # can be None
        scoring = tcfg.get("scoring", "f1")
        cv = int(tcfg.get("cv", 5))
        n_jobs = int(tcfg.get("n_jobs", -1))
        verbosity = int(tcfg.get("verbosity", 2))
        random_state = int(tcfg.get("random_state", self.random_state))
        export_best = bool(tcfg.get("export_best_pipeline", False))
        export_path = str(tcfg.get("export_path", "pipelines/tpot_best_pipeline.py"))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        tpot = TPOTClassifier(
            generations=generations,
            population_size=population_size,
            max_time_mins=max_time_mins,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbosity=verbosity,
            random_state=random_state,
        )

        start = time.time()
        if self.mlflow_enabled:
            with mlflow.start_run(run_name=f"tpot_{name}"):
                self._log_mlflow_params(
                    {
                        "generations": generations,
                        "population_size": population_size,
                        "max_time_mins": max_time_mins,
                        "scoring": scoring,
                        "cv": cv,
                        "n_jobs": n_jobs,
                        "random_state": random_state,
                    }
                )
                tpot.fit(X_tr, y_tr)
                elapsed = time.time() - start
                score = tpot.score(X_te, y_te)
                mlflow.log_metric("score", float(score))
                exported = self._tpot_export(tpot, export_best, export_path)
                if exported:
                    mlflow.log_artifact(exported)
                return {
                    "name": name,
                    "best_score": score,
                    "elapsed_sec": elapsed,
                    "details": {"method": "tpot", "exported": exported},
                }

        # Without MLflow
        tpot.fit(X_tr, y_tr)
        elapsed = time.time() - start
        score = tpot.score(X_te, y_te)
        exported = self._tpot_export(tpot, export_best, export_path)
        return {
            "name": name,
            "best_score": score,
            "elapsed_sec": elapsed,
            "details": {"method": "tpot", "exported": exported},
        }

    def evaluate(self, X, y, spec: Dict[str, Any], cv_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single pipeline spec and return summary dict."""

        # AutoML branch: TPOT
        if spec.get("automl"):
            a = spec["automl"]
            if a.get("library") == "tpot":
                name = a.get("name", "tpot")
                tcfg = a.get("tpot") or {}
                return self._evaluate_tpot(X, y, name, tcfg)
            # Future: other AutoML libraries can be added here
            raise ValueError(f"Unknown AutoML library: {a.get('library')}")

        # GridSearch branch (classic)
        fac = PipelineFactory(self.random_state)
        pipe, grid = fac.build(spec)
        folds = int(cv_cfg.get("cv_folds", 5))
        scoring_cfg = cv_cfg.get("scoring")  # e.g., ["f1"] or "f1" or {"f1": "f1"}

        # Resolve scoring/refit robustly
        if scoring_cfg is None:
            scoring = self.DEFAULT_REFIT
            refit = self.DEFAULT_REFIT
        elif isinstance(scoring_cfg, str):
            scoring = scoring_cfg
            refit = scoring_cfg
        elif isinstance(scoring_cfg, (list, tuple)):
            metrics = list(dict.fromkeys(scoring_cfg))
            self.log.info("metrics(resolu)=%s", metrics)
            if len(metrics) == 1:
                metric = next(iter(metrics))  # plus expressif que metrics
                scoring = metric
                refit = cv_cfg.get("refit") or metric
            else:
                scoring = {m: m for m in metrics}
                refit = cv_cfg.get("refit") or next(iter(metrics))
        elif isinstance(scoring_cfg, dict):
            scoring = scoring_cfg
            refit = cv_cfg.get("refit") or next(iter(scoring_cfg.keys()))
        else:
            raise ValueError(f"Unsupported scoring config: {type(scoring_cfg)}")

        self.log.info(
            "Resolved scoring=%s refit=%s type=%s", scoring, refit, type(refit).__name__
        )

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)

        gridcv = GridSearchCV(
            pipe,
            grid if grid else {},
            scoring=scoring,
            refit=refit,
            cv=cv,
            n_jobs=-1,
            verbose=0,
            return_train_score=False,
        )

        start = time.time()
        gridcv.fit(X, y)
        elapsed = time.time() - start

        cv_path = os.path.join(
            self.out_dir, f"{self.FILE_PREFIX}{spec.get('name')}_{int(start)}{self.FILE_EXT}"
        )
        pd.DataFrame(gridcv.cv_results_).to_csv(cv_path, index=False)

        if self.mlflow_enabled:
            with mlflow.start_run(run_name=f"grid_{spec.get('name')}"):
                for k, v in (grid or {}).items():
                    mlflow.log_param(k, str(v))
                mlflow.log_metric(f"best_{refit}", float(gridcv.best_score_))
                mlflow.log_param("refit_metric", refit)
                mlflow.log_artifact(cv_path)

        return {
            "name": spec.get("name"),
            "best_params": gridcv.best_params_,
            "best_score": gridcv.best_score_,
            "refit_metric": refit,
            "elapsed_sec": elapsed,
            "cv_results_path": cv_path,
        }
