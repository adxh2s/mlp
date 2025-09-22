from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Union, cast

import pandas as pd
import sklearn.model_selection as sk_ms
from sklearn import metrics as sk_metrics
from sklearn.experimental import enable_halving_search_cv as _enable_halving_search_cv  # type: ignore[reportUnusedImport]
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from src.instrumentation.logger_mixin import LoggerMixin, SupportsGetLogger
from src.modeling.pipelines.factory import PipelineFactory

# Optionnels: TPOT / Dask / LazyPredict / Joblib / TPOT2 utils (imports au niveau module)
try:
    from tpot import TPOTClassifier as TPOTClassifierType  # type: ignore[reportMissingTypeStubs]
except Exception:  # noqa: BLE001
    TPOTClassifierType = None  # type: ignore[assignment]

try:
    from dask.distributed import Client as DaskClient, LocalCluster as DaskLocalCluster  # type: ignore[import]
except Exception:  # noqa: BLE001
    DaskClient = None  # type: ignore[assignment]
    DaskLocalCluster = None  # type: ignore[assignment]

try:
    from lazypredict.Supervised import LazyClassifier as LazyClassifierType  # type: ignore[import]
except Exception:  # noqa: BLE001
    LazyClassifierType = None  # type: ignore[assignment]

try:
    from joblib import dump as joblib_dump  # type: ignore[import]
except Exception:  # noqa: BLE001
    joblib_dump = None  # type: ignore[assignment]

try:
    # Présent dans TPOT2
    from tpot2.tpot_estimator.estimator_utils import (  # type: ignore[import]
        apply_make_pipeline as tpot2_apply_make_pipeline,
    )
except Exception:  # noqa: BLE001
    tpot2_apply_make_pipeline = None  # type: ignore[assignment]

"""
Évaluateur de pipelines:
- Optimisation via GridSearchCV / RandomizedSearchCV / Halving* (cv.type).
- AutoML: TPOT (TPOT1/TPOT2 compatibles) et LazyClassifier (leaderboard CSV).
- Dask: LocalCluster optionnel pour TPOT2, avec attente des workers et repli mono-processus.
"""

# =========================
# Constantes (module)
# =========================
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
CV_CV_FOLDS = "cv_folds"
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
LAZY_SAVE_TABLE = "save_table"
LAZY_DEFAULT_CSV = "lazy_results.csv"

# Typages utiles
ScoringType = Union[str, Callable[..., float], dict[str, Any], list[str]]
RefitType = Union[bool, str, Callable[..., Any]]

# Halving* via getattr pour éviter les symboles inconnus dans certains stubs
HalvingGridSearchCV = getattr(sk_ms, "HalvingGridSearchCV", None)
HalvingRandomSearchCV = getattr(sk_ms, "HalvingRandomSearchCV", None)

# Marquer l'import d'activation Halving comme utilisé (évite F401)
_HALVING_IMPORT_USED = bool(_enable_halving_search_cv)


class _F1Weighted(Protocol):
    def __call__(self, y_true: Any, y_pred: Any, *, average: Literal["weighted"]) -> float: ...


def _as_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj)
    if hasattr(obj, "items"):
        try:
            return {str(k): v for k, v in cast(dict[Any, Any], obj).items()}  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(LOGGER_NAME).debug("as_mapping failed: %s", exc)
            return {}
    return {}


def _f1_weighted() -> Callable[[Any, Any], float]:
    avg: Literal["weighted"] = "weighted"
    f1w: _F1Weighted = cast(_F1Weighted, sk_metrics.f1_score)  # type: ignore[reportUnknownMemberType]

    def _metric(y_true: Any, y_pred: Any) -> float:
        return float(f1w(y_true, y_pred, average=avg))

    return _metric


def _tts_df(
    x: pd.DataFrame, y: pd.Series, *, test_size: float, random_state: int, stratify: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    tts = cast(
        Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        sk_ms.train_test_split,  # type: ignore[reportUnknownMemberType]
    )
    return tts(x, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _safe_close(obj: Any, warn: Callable[[str], None], what: str) -> None:
    """Ferme proprement un objet (Client/Cluster) sans déclencher d'alertes de typage."""
    try:
        meth = getattr(obj, "close", None)
        if callable(meth):
            _ = meth()  # peut retourner un awaitable; on ignore le retour volontairement
    except Exception as exc:  # noqa: BLE001
        warn(f"Fermeture {what} a échoué: {exc}")


class PipelineEvaluator(LoggerMixin):
    # =========================
    # Constantes de classe
    # =========================
    # Logger / application
    C_LOGGER_NAME = "mlp.modeling.pipelines.evaluator"
    C_DEFAULT_REFIT = "f1"
    C_FILE_PREFIX = "cv_"
    C_FILE_EXT = ".csv"
    C_DEFAULT_PIPELINE_NAME = "pipeline"

    # CV keys
    K_CV = "cv"
    K_TYPE = "type"
    K_SCORING = "scoring"
    K_REFIT = "refit"
    K_N_SPLITS = "n_splits"
    K_CV_FOLDS = "cv_folds"
    K_SHUFFLE = "shuffle"
    K_RANDOM_STATE = "random_state"
    K_N_JOBS = "n_jobs"
    K_VERBOSE = "verbose"
    K_RETURN_TRAIN_SCORE = "return_train_score"
    K_ERROR_SCORE = "error_score"
    K_N_ITER = "n_iter"
    K_FACTOR = "factor"

    # Search types
    V_SEARCH_GRID = "grid"
    V_SEARCH_RANDOM = "random"
    V_SEARCH_HALVING_GRID = "halving_grid"
    V_SEARCH_HALVING_RANDOM = "halving_random"

    # AutoML keys
    K_AUTOML = "automl"
    K_AUTOML_LIB = "library"
    K_AUTOML_NAME = "name"

    # TPOT keys/values
    V_LIB_TPOT = "tpot"
    K_TPOT = "tpot"
    K_TPOT_GENERATIONS = "generations"
    K_TPOT_POP_SIZE = "population_size"
    K_TPOT_SCORING = "scoring"
    K_TPOT_CV = "cv"
    K_TPOT_N_JOBS = "n_jobs"
    K_TPOT_EXPORT_BEST = "export_best_pipeline"
    K_TPOT_EXPORT_PATH = "export_path"
    V_TPOT_DEFAULT_EXPORT = "tpot_best_pipeline.py"
    V_F1_WEIGHTED = "f1_weighted"
    K_TPOT_VERBOSE2 = "verbose"
    K_TPOT_VERBOSITY1 = "verbosity"
    K_TPOT_PREPROCESSING = "preprocessing"

    # Dask keys/values
    K_DASK_USE = "use_dask"
    K_DASK_N_WORKERS = "n_workers"
    K_DASK_THREADS_PER = "threads_per_worker"
    K_DASK_MEM_LIMIT = "memory_limit"
    K_DASK_WAIT_WORKERS = "wait_for_workers"
    K_DASK_WAIT_TIMEOUT = "wait_timeout_s"
    V_DASK_DASHBOARD = ":8787"

    # LazyPredict keys/values
    V_LIB_LAZY_1 = "lazypredict"
    V_LIB_LAZY_2 = "lazy"
    K_LAZY = "lazy"
    K_LAZY_TEST_SIZE = "test_size"
    K_LAZY_VERBOSE = "verbose"
    K_LAZY_TOP_N = "top_n"
    K_LAZY_TABLE_PATH = "table_path"
    K_LAZY_SAVE_TABLE = "save_table"
    V_LAZY_DEFAULT_CSV = "lazy_results.csv"
    V_LAZY_SCORE_COLS = ("F1 Score", "Accuracy", "ROC AUC", "Balanced Accuracy")

    # Messages
    MSG_UNSUPPORTED_CV = "Unsupported cv.type="
    MSG_DASK_UNAVAILABLE = "Dask indisponible; exécution TPOT sans client."
    MSG_DASK_FAIL = "Échec Dask LocalCluster/Client: "
    MSG_DASK_CLEAN_FAIL = "Nettoyage Dask a échoué: "
    MSG_TPOT_INIT_INCOMPAT = "TPOT init incompatible avec "
    MSG_TPOT_INIT_ERR = "TPOT init erreur: "
    MSG_TPOT_PARAMS_INCOMPAT = "Incompatibilité TPOT paramètres: "
    MSG_LAZY_UNAVAILABLE = "LazyClassifier non disponible"
    MSG_SCORE_READ_FAIL = "Lecture score '{}' impossible: "
    MSG_DASK_CLOSE_FAIL = "Fermeture Dask a échoué: "
    MSG_EXPORT_FAIL = "Export TPOT a échoué: "
    MSG_EXPORT_TPOT2_FAIL = "Export TPOT2 indisponible: "
    MSG_TPOT_NO_EXPORT = "TPOT sans export ni fitted_pipeline_; aucun artefact exporté."

    def __init__(
        self,
        out_dir: str,
        random_state: int = 42,
        mlflow_enabled: bool = False,
        experiment: str = "mlp-experiments",
        logger_manager: SupportsGetLogger | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.random_state = random_state
        self.mlflow_enabled = mlflow_enabled
        self.experiment = experiment
        self.LOGGER_NAME = self.C_LOGGER_NAME
        self._py_logger = logging.getLogger(self.C_LOGGER_NAME)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if logger_manager:
            self._init_logger(logger_manager)

    def _warn(self, msg: str) -> None:
        try:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.warning(msg)
            else:
                self._py_logger.warning(msg)
        except Exception as exc:  # noqa: BLE001
            self._py_logger.debug("warning failed: %s", exc)

    @staticmethod
    def _cv(cv_cfg: dict[str, Any]) -> sk_ms.StratifiedKFold:
        return sk_ms.StratifiedKFold(
            n_splits=int(cv_cfg.get(CV_N_SPLITS, cv_cfg.get(CV_CV_FOLDS, 5))),
            shuffle=bool(cv_cfg.get(CV_SHUFFLE, True)),
            random_state=int(cv_cfg.get(CV_RANDOM_STATE, 0)),
        )

    @staticmethod
    def _scoring_and_refit(cv_cfg: dict[str, Any]) -> tuple[ScoringType, RefitType]:
        scoring: ScoringType = cast(ScoringType, cv_cfg.get(CV_SCORING, DEFAULT_REFIT))
        refit_default: RefitType = DEFAULT_REFIT  # type: ignore[assignment]
        refit: RefitType = cast(RefitType, cv_cfg.get(CV_REFIT, scoring if isinstance(scoring, str) else refit_default))
        return scoring, refit

    @staticmethod
    def _sanitize_space(pipe: Pipeline, space: dict[str, Any] | None) -> dict[str, Any]:
        if not space:
            return {}
        valid: dict[str, Any] = {}
        steps_dict: dict[str, Any] = cast(dict[str, Any], pipe.named_steps)
        for k, v in space.items():
            if "__" not in k:
                valid[k] = v
                continue
            step, _ = k.split("__", 1)
            est = steps_dict.get(step)
            if est is not None and not isinstance(est, str):
                valid[k] = v
        return valid

    def _search(  # noqa: PLR0913
        self,
        cv_type: str,
        estimator: Any,
        scoring: ScoringType,
        refit: RefitType,
        cv: Any,
        grid: dict[str, list[Any]],
        dists: dict[str, Any],
        cv_cfg: dict[str, Any],
    ) -> Any:
        n_jobs = int(cv_cfg.get(self.K_N_JOBS, -1))
        verbose = int(cv_cfg.get(self.K_VERBOSE, 1))
        return_train_score = bool(cv_cfg.get(self.K_RETURN_TRAIN_SCORE, True))
        error_score = cast(str | float, cv_cfg.get(self.K_ERROR_SCORE, "raise"))

        if cv_type == self.V_SEARCH_GRID:
            return sk_ms.GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        if cv_type == self.V_SEARCH_RANDOM:
            return sk_ms.RandomizedSearchCV(
                estimator=estimator,
                param_distributions=dists if dists else grid,
                n_iter=int(cv_cfg.get(self.K_N_ITER, 30)),
                random_state=int(cv_cfg.get(self.K_RANDOM_STATE, 0)),
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        if cv_type == self.V_SEARCH_HALVING_GRID and HalvingGridSearchCV is not None:
            return HalvingGridSearchCV(
                estimator=estimator,
                param_grid=grid,
                factor=int(cv_cfg.get(self.K_FACTOR, 2)),
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        if cv_type == self.V_SEARCH_HALVING_RANDOM and HalvingRandomSearchCV is not None:
            return HalvingRandomSearchCV(
                estimator=estimator,
                param_distributions=dists if dists else grid,
                factor=int(cv_cfg.get(self.K_FACTOR, 2)),
                random_state=int(cv_cfg.get(self.K_RANDOM_STATE, 0)),
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        msg = f"{self.MSG_UNSUPPORTED_CV}{cv_type}"
        raise ValueError(msg)

    # -------------------------
    # Dask helpers
    # -------------------------

    def _make_dask_client(self, tcfg: dict[str, Any]) -> tuple[Any | None, Any | None]:
        if not bool(tcfg.get(self.K_DASK_USE, False)):
            return None, None
        if DaskClient is None or DaskLocalCluster is None:
            self._warn(self.MSG_DASK_UNAVAILABLE)
            return None, None

        n_workers = int(tcfg.get(self.K_DASK_N_WORKERS, 0))
        threads_per_worker = int(tcfg.get(self.K_DASK_THREADS_PER, 1))
        memory_limit = str(tcfg.get(self.K_DASK_MEM_LIMIT, "2GB"))
        wait_for_workers = int(tcfg.get(self.K_DASK_WAIT_WORKERS, max(n_workers, 1)))
        wait_timeout_s = float(tcfg.get(self.K_DASK_WAIT_TIMEOUT, 30))

        if n_workers <= 0:
            return None, None

        client = None
        cluster = None
        try:
            cluster = DaskLocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                dashboard_address=self.V_DASK_DASHBOARD,
            )
            client = DaskClient(cluster)
            client.wait_for_workers(wait_for_workers, timeout=wait_timeout_s)
            return client, cluster
        except Exception as exc:  # noqa: BLE001
            self._warn(f"{self.MSG_DASK_FAIL}{exc}")
            try:
                if client is not None:
                    _safe_close(client, self._warn, "DaskClient")
                if cluster is not None:
                    _safe_close(cluster, self._warn, "DaskLocalCluster")
            except Exception as exc2:  # noqa: BLE001
                self._warn(f"{self.MSG_DASK_CLEAN_FAIL}{exc2}")
            return None, None

    # -------------------------
    # TPOT helpers
    # -------------------------

    def _tpot_trials(
        self, tcfg: dict[str, Any], client: Any | None, safe_cv: int
    ) -> list[dict[str, Any]]:
        scoring_raw = tcfg.get(self.K_TPOT_SCORING, self.C_DEFAULT_REFIT)
        generations = int(tcfg.get(self.K_TPOT_GENERATIONS, 5))
        population_size = int(tcfg.get(self.K_TPOT_POP_SIZE, 50))
        verbose2 = int(tcfg.get(self.K_TPOT_VERBOSE2, 2))
        verbosity1 = int(tcfg.get(self.K_TPOT_VERBOSITY1, 2))
        n_jobs = int(tcfg.get(self.K_TPOT_N_JOBS, -1)) if client is None else 1

        trials: list[dict[str, Any]] = []

        # TPOT2 (client + scorers + verbose + preprocessing)
        scoring2 = (
            self.V_F1_WEIGHTED
            if isinstance(scoring_raw, str) and scoring_raw.lower() in ("f1", self.V_F1_WEIGHTED)
            else scoring_raw
        )
        scorers = [scoring2] if isinstance(scoring2, str) else scoring2
        if client is not None:
            trials.append(
                {
                    "cv": safe_cv,
                    "random_state": self.random_state,
                    "verbose": verbose2,
                    "scorers": scorers,
                    "scorers_weights": [1] * len(cast(list[Any], scorers)),
                    "client": client,
                    "generations": generations,
                    "population_size": population_size,
                    "preprocessing": bool(tcfg.get(self.K_TPOT_PREPROCESSING, True)),
                }
            )

        # TPOT1 (scoring + verbose)
        trials.append(
            {
                "cv": safe_cv,
                "n_jobs": n_jobs,
                "random_state": self.random_state,
                "verbose": verbose2,
                "scoring": scoring_raw,
                "generations": generations,
                "population_size": population_size,
            }
        )

        # TPOT1 legacy (scoring + verbosity)
        trials.append(
            {
                "cv": safe_cv,
                "n_jobs": n_jobs,
                "random_state": self.random_state,
                "verbosity": verbosity1,
                "scoring": scoring_raw,
                "generations": generations,
                "population_size": population_size,
            }
        )
        return trials

    def _instantiate_tpot(self, trials: list[dict[str, Any]]) -> tuple[Any | None, Exception | None]:
        if TPOTClassifierType is None:
            return None, ModuleNotFoundError("tpot non installé")
        last_exc: Exception | None = None
        for kw in trials:
            try:
                return TPOTClassifierType(**kw), None  # type: ignore[call-arg]
            except TypeError as exc:  # mauvais set de paramètres -> essayer suivant
                last_exc = exc
                self._warn(f"{self.MSG_TPOT_INIT_INCOMPAT}{list(kw.keys())}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._warn(f"{self.MSG_TPOT_INIT_ERR}{exc}")
                continue
        return None, last_exc

    # -------------------------
    # AutoML runners
    # -------------------------

    def _maybe_run_tpot(  # noqa: C901, PLR0912, PLR0915
        self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series
    ) -> dict[str, Any] | None:
        automl = _as_mapping(spec.get(self.K_AUTOML))
        if str(automl.get(self.K_AUTOML_LIB, "")).lower() != self.V_LIB_TPOT:
            return None

        tcfg = _as_mapping(automl.get(self.K_TPOT))
        client = None
        cluster = None
        try:
            client, cluster = self._make_dask_client(tcfg)

            # Bornage de cv par la minorité de classe (faisabilité StratifiedKFold)
            requested_cv = int(tcfg.get(self.K_TPOT_CV, 5))
            min_per_class = int(pd.Series(y).value_counts().min())
            safe_cv = max(2, min(requested_cv, min_per_class))

            trials = self._tpot_trials(tcfg, client, safe_cv)
            tpot, err = self._instantiate_tpot(trials)
            if tpot is None:
                msg = f"{self.MSG_TPOT_PARAMS_INCOMPAT}{err}"
                self._warn(msg)
                return {
                    "name": automl.get(self.K_AUTOML_NAME, self.V_LIB_TPOT),
                    "best_score": None,
                    "error": msg,
                    "artifacts": [],
                    "cv_results_path": None,
                }

            t0 = time.time()
            tpot.fit(x, y)  # type: ignore[attr-defined]
            dur = time.time() - t0

            export = bool(tcfg.get(self.K_TPOT_EXPORT_BEST, False))
            export_path = self.out_dir / str(tcfg.get(self.K_TPOT_EXPORT_PATH, self.V_TPOT_DEFAULT_EXPORT))
            artifacts: list[str] = []
            if export:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # TPOT 1.x: export code Python si disponible
                    if hasattr(tpot, "export"):
                        tpot.export(str(export_path))  # type: ignore[attr-defined]
                        artifacts = [str(export_path)]
                    else:
                        # Fallback universel: dump joblib du pipeline entraîné
                        pipe: Any = getattr(tpot, "fitted_pipeline_", None)
                        if pipe is not None and joblib_dump is not None:
                            pkl_path = export_path.with_suffix(".pkl") if export_path.suffix == ".py" else export_path
                            joblib_dump(pipe, str(pkl_path))
                            artifacts = [str(pkl_path)]
                        elif tpot2_apply_make_pipeline is not None and joblib_dump is not None:
                            ind: Any = getattr(tpot, "best_individual_", None)
                            if ind is not None:
                                skl_pipe: Pipeline = cast(
                                    Pipeline,
                                    tpot2_apply_make_pipeline(ind, preprocessing_pipeline=None, export_graphpipeline=False),
                                )
                                pkl_path = export_path.with_suffix(".pkl")
                                joblib_dump(skl_pipe, str(pkl_path))
                                artifacts = [str(pkl_path)]
                            else:
                                self._warn(self.MSG_TPOT_NO_EXPORT)
                        else:
                            self._warn(self.MSG_TPOT_NO_EXPORT)
                except Exception as exc:  # noqa: BLE001
                    self._warn(f"{self.MSG_EXPORT_FAIL}{exc}")

            # Calcul du score robuste (TPOT1: score; TPOT2: pipeline + scorer)
            scoring_raw = tcfg.get(self.K_TPOT_SCORING, self.C_DEFAULT_REFIT)
            scoring_name = (
                self.V_F1_WEIGHTED
                if isinstance(scoring_raw, str) and scoring_raw.lower() in ("f1", self.V_F1_WEIGHTED)
                else (str(scoring_raw) if isinstance(scoring_raw, str) else self.V_F1_WEIGHTED)
            )
            if hasattr(tpot, "score"):
                best = float(tpot.score(x, y))  # type: ignore[attr-defined]
            else:
                pipe_for_score: Any = getattr(tpot, "fitted_pipeline_", None)
                if pipe_for_score is None:
                    best = 0.0
                else:
                    try:
                        scorer = get_scorer(scoring_name)
                        best = float(scorer(pipe_for_score, x, y))
                    except Exception:
                        y_pred = pipe_for_score.predict(x)
                        best = float(sk_metrics.f1_score(y, y_pred, average="weighted"))

            return {
                "name": automl.get(self.K_AUTOML_NAME, self.V_LIB_TPOT),
                "best_score": best,
                "duration_sec": dur,
                "artifacts": artifacts,
                "cv_results_path": None,
                "best_params": {},
            }
        finally:
            try:
                if client is not None:
                    _safe_close(client, self._warn, "DaskClient")
                if cluster is not None:
                    _safe_close(cluster, self._warn, "DaskLocalCluster")
            except Exception as exc:  # noqa: BLE001
                self._warn(f"{self.MSG_DASK_CLOSE_FAIL}{exc}")

    def _maybe_run_lazy(
        self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series
    ) -> dict[str, Any] | None:
        automl = _as_mapping(spec.get(self.K_AUTOML))
        lib = str(automl.get(self.K_AUTOML_LIB, "")).lower()
        if lib not in {self.V_LIB_LAZY_1, self.V_LIB_LAZY_2}:
            return None
        if LazyClassifierType is None:
            msg = self.MSG_LAZY_UNAVAILABLE
            self._warn(msg)
            return {
                "name": automl.get(self.K_AUTOML_NAME, self.V_LIB_LAZY_1),
                "best_score": None,
                "error": msg,
                "artifacts": [],
                "cv_results_path": None,
            }

        lcfg = _as_mapping(automl.get(self.K_LAZY))
        test_size = float(lcfg.get(self.K_LAZY_TEST_SIZE, 0.2))
        x_train, x_test, y_train, y_test = _tts_df(
            x, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        include_raw = lcfg.get("include")
        classifiers = self._resolve_classifiers(include_raw if include_raw else None)

        lazy_kwargs: dict[str, Any] = {
            "verbose": int(lcfg.get(self.K_LAZY_VERBOSE, 0)),
            "ignore_warnings": True,
            "custom_metric": self._resolve_metric(lcfg.get("custom_metric")),
        }
        if classifiers:
            lazy_kwargs["classifiers"] = classifiers

        clf = LazyClassifierType(**lazy_kwargs)
        t0 = time.time()
        models_df, _unused = cast(tuple[pd.DataFrame, Any], clf.fit(x_train, x_test, y_train, y_test))
        dur = time.time() - t0

        top_n = int(lcfg.get(self.K_LAZY_TOP_N, 25))
        models_df = cast(pd.DataFrame, models_df)
        models_df = models_df.head(top_n).copy()

        table_path = self.out_dir / str(lcfg.get(self.K_LAZY_TABLE_PATH, self.V_LAZY_DEFAULT_CSV))
        if bool(lcfg.get(self.K_LAZY_SAVE_TABLE, True)):
            table_path.parent.mkdir(parents=True, exist_ok=True)
            models_df.to_csv(table_path, index=True)

        best_score: float | None = None
        for col in self.V_LAZY_SCORE_COLS:
            if col in models_df.columns and len(models_df) > 0:
                try:
                    val: Any = models_df.iloc[0][col]
                    best_score = float(val)
                    break
                except Exception as exc:  # noqa: BLE001
                    self._warn(self.MSG_SCORE_READ_FAIL.format(col) + f"{exc}")
                    continue

        return {
            "name": automl.get(self.K_AUTOML_NAME, self.V_LIB_LAZY_1),
            "best_score": best_score,
            "duration_sec": dur,
            "artifacts": [str(table_path)] if bool(lcfg.get(self.K_LAZY_SAVE_TABLE, True)) else [],
            "cv_results_path": str(table_path) if bool(lcfg.get(self.K_LAZY_SAVE_TABLE, True)) else None,
            "best_params": {},
        }

    # -------------------------
    # Résolution d’algorithmes sklearn pour LazyPredict
    # -------------------------

    @staticmethod
    def _resolve_metric(metric_spec: Any) -> Callable[..., float] | None:
        if metric_spec in (None, "", "null"):
            return None
        if callable(metric_spec):
            return cast(Callable[..., float], metric_spec)
        if isinstance(metric_spec, str) and metric_spec.lower().startswith("f1"):
            return _f1_weighted()
        return None

    @staticmethod
    def _resolve_sklearn_class(short_name: str) -> type | None:
        candidates = [
            "sklearn.ensemble",
            "sklearn.linear_model",
            "sklearn.svm",
            "sklearn.neighbors",
            "sklearn.tree",
            "sklearn.naive_bayes",
            "sklearn.neural_network",
            "sklearn.discriminant_analysis",
            "sklearn.gaussian_process",
            "sklearn.semi_supervised",
        ]
        for mod in candidates:
            try:
                module = __import__(mod, fromlist=[short_name])
                klass = getattr(module, short_name, None)
                if isinstance(klass, type):
                    return klass
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(LOGGER_NAME).debug("Resolve class failed in module %s: %s", mod, exc)
                continue
        return None

    @staticmethod
    def _resolve_class(path_or_name: Any) -> type | None:
        if isinstance(path_or_name, type):
            return path_or_name
        if not isinstance(path_or_name, str):
            return None
        name = path_or_name.strip()
        if "." in name:
            try:
                module_path, cls_name = name.rsplit(".", 1)
                module = __import__(module_path, fromlist=[cls_name])
                klass = getattr(module, cls_name, None)
                return klass if isinstance(klass, type) else None
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(LOGGER_NAME).debug("Import class failed for %s: %s", name, exc)
                return None
        return PipelineEvaluator._resolve_sklearn_class(name)

    @staticmethod
    def _resolve_classifiers(items: Sequence[Any] | None) -> list[type] | None:
        if not items:
            return None
        resolved: list[type] = []
        for it in items:
            klass = PipelineEvaluator._resolve_class(it)
            if isinstance(klass, type):
                resolved.append(klass)
        return resolved or None

    # -------------------------
    # Entrée principale
    # -------------------------

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        spec: dict[str, Any],
        cv_cfg: dict[str, Any],
        global_policy: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._maybe_run_tpot(spec, x, y)
        if out is not None:
            return out
        out = self._maybe_run_lazy(spec, x, y)
        if out is not None:
            return out

        pipe, grid, dists = PipelineFactory.build(spec, global_policy)

        grid = self._sanitize_space(pipe, grid)
        dists = self._sanitize_space(pipe, dists)

        scoring, refit = self._scoring_and_refit(cv_cfg)
        cv = self._cv(cv_cfg)
        cv_type = str(cv_cfg.get(self.K_TYPE, self.V_SEARCH_GRID))
        search = self._search(cv_type, pipe, scoring, refit, cv, grid, dists, cv_cfg)

        t0 = time.time()
        search.fit(x, y)
        duration = time.time() - t0

        cv_path = self.out_dir / f"{self.C_FILE_PREFIX}{spec.get('name', self.C_DEFAULT_PIPELINE_NAME)}{self.C_FILE_EXT}"
        pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)

        return {
            "name": spec.get("name", self.C_DEFAULT_PIPELINE_NAME),
            "best_score": float(search.best_score_),
            "best_params": dict(search.best_params_),
            "duration_sec": duration,
            "cv_results_path": str(cv_path),
            "artifacts": [str(cv_path)],
        }
