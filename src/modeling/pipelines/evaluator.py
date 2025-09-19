from __future__ import annotations

"""
Évaluateur de pipelines:
- Optimisation via GridSearchCV / RandomizedSearchCV / Halving* (cv.type).
- AutoML: TPOT et LazyClassifier (leaderboard CSV).
- Export des résultats en CSV et intégration DI logger/messages.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src.instrumentation.logger_mixin import LoggerMixin, SupportsGetLogger
from src.modeling.pipelines.factory import PipelineFactory

# =========================
# Constantes (clés & valeurs)
# =========================

# Logger / sortie
LOGGER_NAME = "mlp.modeling.pipelines.evaluator"
DEFAULT_REFIT = "f1"
FILE_PREFIX = "cv_"
FILE_EXT = ".csv"

# Clés section CV
CV_KEY = "cv"
CV_TYPE = "type"
CV_SCORING = "scoring"
CV_REFIT = "refit"
CV_N_SPLITS = "n_splits"
CV_CV_FOLDS = "cv_folds"      # alias
CV_SHUFFLE = "shuffle"
CV_RANDOM_STATE = "random_state"
CV_N_JOBS = "n_jobs"
CV_VERBOSE = "verbose"
CV_RETURN_TRAIN_SCORE = "return_train_score"
CV_ERROR_SCORE = "error_score"
CV_N_ITER = "n_iter"
CV_FACTOR = "factor"

# Types de recherche
SEARCH_GRID = "grid"
SEARCH_RANDOM = "random"
SEARCH_HALVING_GRID = "halving_grid"
SEARCH_HALVING_RANDOM = "halving_random"

# Clés AutoML génériques
AUTO_KEY = "automl"
AUTO_LIB = "library"
AUTO_NAME = "name"

# TPOT
LIB_TPOT = "tpot"
TPOT_KEY = "tpot"
TPOT_GENERATIONS = "generations"
TPOT_POP_SIZE = "population_size"
TPOT_SCORING = "scoring"
TPOT_CV = "cv"
TPOT_N_JOBS = "n_jobs"
TPOT_EXPORT_BEST = "export_best_pipeline"
TPOT_EXPORT_PATH = "export_path"
TPOT_DEFAULT_EXPORT = "tpot_best_pipeline.py"

# LazyPredict
LIB_LAZY_1 = "lazypredict"
LIB_LAZY_2 = "lazy"
LAZY_KEY = "lazy"
LAZY_TEST_SIZE = "test_size"
LAZY_VERBOSE = "verbose"
LAZY_TOP_N = "top_n"
LAZY_TABLE_PATH = "table_path"
LAZY_DEFAULT_CSV = "lazy_results.csv"

# =========================
# Implémentation
# =========================


class PipelineEvaluator(LoggerMixin):
    """Évalue un spec pipeline avec l’algorithme d’optimisation demandé ou un backend AutoML dédié."""

    def __init__(
        self,
        out_dir: str,
        random_state: int = 42,
        mlflow_enabled: bool = False,
        experiment: str = "mlp-experiments",
        logger_manager: SupportsGetLogger | None = None,
    ) -> None:
        """
        Args:
            out_dir: Répertoire de sortie pour artefacts (CSV, exports AutoML).
            random_state: Graine pour CV / algos stochastiques.
            mlflow_enabled: Réservé (tracking optionnel).
            experiment: Nom d’expérience MLflow.
            logger_manager: Gestionnaire de logger injecté par l’orchestrateur.
        """
        self.out_dir = Path(out_dir)
        self.random_state = random_state
        self.mlflow_enabled = mlflow_enabled
        self.experiment = experiment
        self.LOGGER_NAME = LOGGER_NAME
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if logger_manager:
            self._init_logger(logger_manager)

    @staticmethod
    def _cv(cv_cfg: Dict[str, Any]) -> StratifiedKFold:
        """Construit un StratifiedKFold depuis la section cv du YAML."""
        return StratifiedKFold(
            n_splits=int(cv_cfg.get(CV_N_SPLITS, cv_cfg.get(CV_CV_FOLDS, 5))),
            shuffle=bool(cv_cfg.get(CV_SHUFFLE, True)),
            random_state=int(cv_cfg.get(CV_RANDOM_STATE, 0)),
        )

    @staticmethod
    def _scoring_and_refit(cv_cfg: Dict[str, Any]) -> Tuple[Any, Any]:
        """Extrait scoring/refit; par défaut refit = scoring si scoring est str."""
        scoring = cv_cfg.get(CV_SCORING, DEFAULT_REFIT)
        refit = cv_cfg.get(CV_REFIT, scoring if isinstance(scoring, str) else DEFAULT_REFIT)
        return scoring, refit

    def _search(self, cv_type: str, estimator, scoring, refit, cv, grid, dists, cv_cfg: Dict[str, Any]):
        """Construit l’objet de recherche d’hyperparamètres selon cv.type."""
        common = dict(
            scoring=scoring,
            refit=refit,
            cv=cv,
            n_jobs=int(cv_cfg.get(CV_N_JOBS, -1)),
            verbose=int(cv_cfg.get(CV_VERBOSE, 1)),
            return_train_score=bool(cv_cfg.get(CV_RETURN_TRAIN_SCORE, True)),
            error_score=cv_cfg.get(CV_ERROR_SCORE, "raise"),
        )
        if cv_type == SEARCH_GRID:
            return GridSearchCV(estimator, param_grid=grid, **common)
        if cv_type == SEARCH_RANDOM:
            return RandomizedSearchCV(
                estimator,
                param_distributions=dists if dists else grid,
                n_iter=int(cv_cfg.get(CV_N_ITER, 30)),
                random_state=int(cv_cfg.get(CV_RANDOM_STATE, self.random_state)),
                **common,
            )
        if cv_type == SEARCH_HALVING_GRID:
            return HalvingGridSearchCV(
                estimator,
                param_grid=grid,
                factor=int(cv_cfg.get(CV_FACTOR, 2)),
                **common,
            )
        if cv_type == SEARCH_HALVING_RANDOM:
            return HalvingRandomSearchCV(
                estimator,
                param_distributions=dists if dists else grid,
                factor=int(cv_cfg.get(CV_FACTOR, 2)),
                random_state=int(cv_cfg.get(CV_RANDOM_STATE, self.random_state)),
                **common,
            )
        msg = f"Unsupported cv.type={cv_type}"
        raise ValueError(msg)

    def _maybe_run_tpot(self, spec: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Optional[Dict[str, Any]]:
        """Exécute TPOT si automl.library == 'tpot' et retourne un résumé."""
        automl = spec.get(AUTO_KEY) or {}
        if str(automl.get(AUTO_LIB, "")).lower() != LIB_TPOT:
            return None
        try:
            from tpot import TPOTClassifier  # lazy import
        except Exception as exc:  # noqa: BLE001
            msg = f"TPOT non disponible: {exc}"
            return {
                "name": automl.get(AUTO_NAME, LIB_TPOT),
                "best_score": None,
                "error": msg,
                "artifacts": [],
                "cv_results_path": None,
            }

        tcfg = automl.get(TPOT_KEY) or {}
        tpot = TPOTClassifier(
            generations=int(tcfg.get(TPOT_GENERATIONS, 5)),
            population_size=int(tcfg.get(TPOT_POP_SIZE, 50)),
            scoring=tcfg.get(TPOT_SCORING, DEFAULT_REFIT),
            cv=int(tcfg.get(TPOT_CV, 5)),
            n_jobs=int(tcfg.get(TPOT_N_JOBS, -1)),
            random_state=self.random_state,
            verbosity=2,
        )
        t0 = time.time()
        tpot.fit(X, y)
        dur = time.time() - t0
        export = bool(tcfg.get(TPOT_EXPORT_BEST, False))
        export_path = self.out_dir / str(tcfg.get(TPOT_EXPORT_PATH, TPOT_DEFAULT_EXPORT))
        if export:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            tpot.export(str(export_path))
        return {
            "name": automl.get(AUTO_NAME, LIB_TPOT),
            "best_score": float(tpot.score(X, y)),
            "duration_sec": dur,
            "artifacts": [str(export_path)] if export else [],
            "cv_results_path": None,
            "best_params": {},
        }

    def _maybe_run_lazy(self, spec: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Optional[Dict[str, Any]]:
        """Exécute LazyClassifier si automl.library == 'lazypredict'/'lazy' et retourne un leaderboard CSV."""
        automl = spec.get(AUTO_KEY) or {}
        lib = str(automl.get(AUTO_LIB, "")).lower()
        if lib not in {LIB_LAZY_1, LIB_LAZY_2}:
            return None
        try:
            from lazypredict.Supervised import LazyClassifier  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            msg = f"LazyClassifier non disponible: {exc}"
            return {
                "name": automl.get(AUTO_NAME, LIB_LAZY_1),
                "best_score": None,
                "error": msg,
                "artifacts": [],
                "cv_results_path": None,
            }

        lcfg = automl.get(LAZY_KEY) or {}
        test_size = float(lcfg.get(LAZY_TEST_SIZE, 0.2))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        clf = LazyClassifier(
            verbose=int(lcfg.get(LAZY_VERBOSE, 0)),
            ignore_warnings=True,
            custom_metric=None,
            classifiers=None,  # include/exclude non standardisés -> défauts
        )
        t0 = time.time()
        models_df, _ = clf.fit(X_train, X_test, y_train, y_test)
        dur = time.time() - t0

        top_n = int(lcfg.get(LAZY_TOP_N, 25))
        models_df = models_df.head(top_n).copy()

        table_path = self.out_dir / str(lcfg.get(LAZY_TABLE_PATH, LAZY_DEFAULT_CSV))
        table_path.parent.mkdir(parents=True, exist_ok=True)
        models_df.to_csv(table_path, index=True)

        best_score = float(models_df.iloc[0]["Accuracy"]) if "Accuracy" in models_df.columns and len(models_df) > 0 else None
        return {
            "name": automl.get(AUTO_NAME, LIB_LAZY_1),
            "best_score": best_score,
            "duration_sec": dur,
            "artifacts": [str(table_path)],
            "cv_results_path": str(table_path),
            "best_params": {},
        }

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        spec: Dict[str, Any],
        cv_cfg: Dict[str, Any],
        global_policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Route l’évaluation:
        - AutoML: TPOT / LazyClassifier si 'automl' est défini.
        - Sinon: construction du Pipeline via la fabrique, puis recherche selon cv.type.
        """
        # Branches AutoML
        out = self._maybe_run_tpot(spec, X, y)
        if out is not None:
            return out
        out = self._maybe_run_lazy(spec, X, y)
        if out is not None:
            return out

        # Pipeline + grids/distributions
        pipe, grid, dists = PipelineFactory.build(spec, global_policy)
        scoring, refit = self._scoring_and_refit(cv_cfg)
        cv = self._cv(cv_cfg)
        cv_type = str(cv_cfg.get(CV_TYPE, SEARCH_GRID))

        search = self._search(cv_type, pipe, scoring, refit, cv, grid, dists, cv_cfg)

        t0 = time.time()
        search.fit(X, y)
        duration = time.time() - t0

        # Export cv_results_
        cv_path = self.out_dir / f"{FILE_PREFIX}{spec.get('name', 'pipeline')}{FILE_EXT}"
        pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)

        return {
            "name": spec.get("name", "pipeline"),
            "best_score": float(search.best_score_),
            "best_params": dict(search.best_params_),
            "duration_sec": duration,
            "cv_results_path": str(cv_path),
            "artifacts": [str(cv_path)],
        }
