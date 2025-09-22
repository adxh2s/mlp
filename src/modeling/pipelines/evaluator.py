from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.model_selection as sk_ms
from sklearn import metrics as sk_metrics
from sklearn.experimental import enable_halving_search_cv as _enable_halving_search_cv  # noqa: F401
from sklearn.pipeline import Pipeline

from src.instrumentation.logger_mixin import LoggerMixin, SupportsGetLogger
from src.modeling.dl.config import DLConfig
from src.modeling.dl.trainer import train_dense
from src.modeling.pipelines.consts import (
    AUTO_KEY,
    AUTO_LIB,
    AUTO_NAME,
    CV_CV_FOLDS,
    CV_ERROR_SCORE,
    CV_FACTOR,
    CV_N_ITER,
    CV_N_JOBS,
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    CV_REFIT,
    CV_RETURN_TRAIN_SCORE,
    CV_SCORING,
    CV_SHUFFLE,
    CV_TYPE,
    CV_VERBOSE,
    DEFAULT_PIPELINE_NAME,
    DEFAULT_REFIT,
    DL_BEST_SCORE_KEYS,
    F1_WEIGHTED,
    FILE_EXT,
    FILE_PREFIX,
    LAZY_DEFAULT_CSV,
    LAZY_KEY,
    LAZY_SAVE_TABLE,
    LAZY_SCORE_COLS,
    LAZY_TABLE_PATH,
    LAZY_TEST_SIZE,
    LAZY_TOP_N,
    LAZY_VERBOSE,
    LOGGER_NAME,
    MSG_DASK_CLEAN_FAIL,
    MSG_DASK_CLOSE_FAIL,
    MSG_DASK_FAIL,
    MSG_DASK_UNAVAILABLE,
    MSG_EXPORT_FAIL,
    MSG_LAZY_UNAVAILABLE,
    MSG_SCORE_READ_FAIL,
    MSG_TPOT_INIT_ERR,
    MSG_TPOT_INIT_INCOMPAT,
    MSG_TPOT_NO_EXPORT,
    MSG_TPOT_PARAMS_INCOMPAT,
    MSG_UNSUPPORTED_CV,
    SEARCH_GRID,
    SEARCH_HALVING_GRID,
    SEARCH_HALVING_RANDOM,
    SEARCH_RANDOM,
    TPOT_CV,
    TPOT_DEFAULT_EXPORT,
    TPOT_EXPORT_BEST,
    TPOT_EXPORT_PATH,
    TPOT_GENERATIONS,
    TPOT_KEY,
    TPOT_N_JOBS,
    TPOT_POP_SIZE,
    TPOT_PREPROCESSING,
    TPOT_SCORING,
    TPOT_VERBOSE2,
    TPOT_VERBOSITY1,
)
from src.modeling.pipelines.factory import PipelineFactory

# Dépendances optionnelles
try:  # TPOT
    from tpot import TPOTClassifier as TPOTClassifierType  # type: ignore[reportMissingTypeStubs]
except Exception:  # noqa: BLE001
    TPOTClassifierType = None  # type: ignore[assignment]

try:  # Dask
    from dask.distributed import Client as DaskClient  # type: ignore[import]
    from dask.distributed import LocalCluster as DaskLocalCluster
except Exception:  # noqa: BLE001
    DaskClient = None  # type: ignore[assignment]
    DaskLocalCluster = None  # type: ignore[assignment]

try:  # LazyPredict
    from lazypredict.Supervised import LazyClassifier as LazyClassifierType  # type: ignore[import]
except Exception:  # noqa: BLE001
    LazyClassifierType = None  # type: ignore[assignment]

try:  # Joblib
    from joblib import dump as joblib_dump  # type: ignore[import]
except Exception:  # noqa: BLE001
    joblib_dump = None  # type: ignore[assignment]

# Import dynamique de TPOT2 apply_make_pipeline
_tpot2_apply: Callable[..., Pipeline] | None = None
try:
    _tpot2_mod = importlib.import_module("tpot2.tpot_estimator.estimator_utils")
    _tpot2_apply = cast(Callable[..., Pipeline], getattr(_tpot2_mod, "apply_make_pipeline", None))
except Exception:  # noqa: BLE001
    _tpot2_apply = None

# Types utiles
ScoringType = str | Callable[..., float] | dict[str, Any] | list[str]
RefitType = bool | str | Callable[..., Any]

# Halving* via getattr
HalvingGridSearchCV = getattr(sk_ms, "HalvingGridSearchCV", None)
HalvingRandomSearchCV = getattr(sk_ms, "HalvingRandomSearchCV", None)

# Marquer l'import d'activation Halving comme utilisé
_HALVING_IMPORT_USED = bool(_enable_halving_search_cv)


"""
Évaluateur de pipelines ML/DL:
- Sélectionne et exécute ML (sklearn + CV) ou DL (Keras) selon la config déclarative.
- AutoML: TPOT (TPOT1/TPOT2) et LazyPredict, avec export d'artefacts et scores robustes.
- Dask optionnel pour TPOT2, fermeture sûre des ressources, et logs harmonisés.
"""

class _F1Weighted(Protocol):
    def __call__(self, y_true: Any, y_pred: Any, *, average: Literal["weighted"]) -> float: ...


def _get_scorer_safe(name: str | None) -> Callable[[Any, Any, Any], float] | None:
    """Wrapper typé sur sklearn.metrics.get_scorer, retourne (estimator, X, y) -> float ou None."""
    try:
        if not name:
            return None
        return cast(Callable[[Any, Any, Any], float], sk_metrics.get_scorer(name))  # type: ignore[reportUnknownMemberType]
    except Exception:  # noqa: BLE001
        return None


def _as_mapping(obj: Any) -> dict[str, Any]:
    """Copie défensive d’un mapping arbitraire en dict[str, Any]."""
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
    """Scorer F1 weighted simple (pour LazyPredict custom_metric)."""
    avg: Literal["weighted"] = "weighted"
    f1w: _F1Weighted = cast(_F1Weighted, sk_metrics.f1_score)  # type: ignore[reportUnknownMemberType]

    def _metric(y_true: Any, y_pred: Any) -> float:
        return float(f1w(y_true, y_pred, average=avg))

    return _metric


def _tts_df(
    x: pd.DataFrame, y: pd.Series, *, test_size: float, random_state: int, stratify: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split DataFrame/Series avec stratification robuste."""
    tts = cast(
        Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        sk_ms.train_test_split,  # type: ignore[reportUnknownMemberType]
    )
    return tts(x, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _tts_np(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    *,
    test_size: float,
    random_state: int,
    stratify: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Train/test split NDArray avec typage explicite, via cast sur l’API sklearn."""
    tts = cast(
        Callable[
            ...,
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
        ],
        sk_ms.train_test_split,  # type: ignore[reportUnknownMemberType]
    )
    return tts(x, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _safe_close(obj: Any, warn: Callable[[str], None], what: str) -> None:
    """Ferme proprement un objet (Client/Cluster) sans déclencher d’alertes de typage."""
    try:
        meth = getattr(obj, "close", None)
        if callable(meth):
            _ = meth()
    except Exception as exc:  # noqa: BLE001
        warn(f"Fermeture {what} a échoué: {exc}")


class PipelineEvaluator(LoggerMixin):
    """
    Évalue un pipeline ML ou DL selon la configuration déclarative fournie.
    - ML: pipelines sklearn + CV (Grid/Random/Halving), export des cv_results_.csv.
    - DL: modèles Keras séquentiels/fonctionnels via module modeling/dl, export .keras et history.
    """

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
        self.LOGGER_NAME = LOGGER_NAME
        self._py_logger = logging.getLogger(LOGGER_NAME)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if logger_manager:
            self._init_logger(logger_manager)

    def _warn(self, msg: str) -> None:
        """Émet un warning via le logger mixin si présent, sinon logger Python."""
        try:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.warning(msg)
            else:
                self._py_logger.warning(msg)
        except Exception as exc:  # noqa: BLE001
            self._py_logger.debug("warning failed: %s", exc)

    # -------------------------
    # Config CV/scoring
    # -------------------------

    @staticmethod
    def _cv(cv_cfg: dict[str, Any]) -> sk_ms.StratifiedKFold:
        """Construit un StratifiedKFold robuste depuis la config CV."""
        return sk_ms.StratifiedKFold(
            n_splits=int(cv_cfg.get(CV_N_SPLITS, cv_cfg.get(CV_CV_FOLDS, 5))),
            shuffle=bool(cv_cfg.get(CV_SHUFFLE, True)),
            random_state=int(cv_cfg.get(CV_RANDOM_STATE, 0)),
        )

    @staticmethod
    def _scoring_and_refit(cv_cfg: dict[str, Any]) -> tuple[ScoringType, RefitType]:
        """Retourne le couple scoring/refit compatible sklearn, avec défauts sûrs."""
        scoring: ScoringType = cast(ScoringType, cv_cfg.get(CV_SCORING, DEFAULT_REFIT))
        refit_default: RefitType = DEFAULT_REFIT  # type: ignore[assignment]
        refit: RefitType = cast(RefitType, cv_cfg.get(CV_REFIT, scoring if isinstance(scoring, str) else refit_default))
        return scoring, refit

    @staticmethod
    def _sanitize_space(pipe: Pipeline, space: dict[str, Any] | None) -> dict[str, Any]:
        """Filtre l’espace de recherche pour ne garder que les clés valides pour le pipeline."""
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

    # -------------------------
    # Constructeurs de recherche
    # -------------------------

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
        """Construit l’objet de recherche hyperparamétrique selon le type souhaité."""
        n_jobs = int(cv_cfg.get(CV_N_JOBS, -1))
        verbose = int(cv_cfg.get(CV_VERBOSE, 1))
        return_train_score = bool(cv_cfg.get(CV_RETURN_TRAIN_SCORE, True))
        error_score = cast(str | float, cv_cfg.get(CV_ERROR_SCORE, "raise"))

        if cv_type == SEARCH_GRID:
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

        if cv_type == SEARCH_RANDOM:
            return sk_ms.RandomizedSearchCV(
                estimator=estimator,
                param_distributions=dists if dists else grid,
                n_iter=int(cv_cfg.get(CV_N_ITER, 30)),
                random_state=int(cv_cfg.get(CV_RANDOM_STATE, 0)),
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        if cv_type == SEARCH_HALVING_GRID and HalvingGridSearchCV is not None:
            return HalvingGridSearchCV(
                estimator=estimator,
                param_grid=grid,
                factor=int(cv_cfg.get(CV_FACTOR, 2)),
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        if cv_type == SEARCH_HALVING_RANDOM and HalvingRandomSearchCV is not None:
            return HalvingRandomSearchCV(
                estimator=estimator,
                param_distributions=dists if dists else grid,
                factor=int(cv_cfg.get(CV_FACTOR, 2)),
                random_state=int(cv_cfg.get(CV_RANDOM_STATE, 0)),
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=return_train_score,
                error_score=error_score,
            )

        msg = f"{MSG_UNSUPPORTED_CV}{cv_type}"
        raise ValueError(msg)

    # -------------------------
    # Dask helpers
    # -------------------------

    def _make_dask_client(self, tcfg: dict[str, Any]) -> tuple[Any | None, Any | None]:
        """Crée un cluster local Dask si demandé; sinon retourne (None, None)."""
        if not bool(tcfg.get("use_dask", False)):
            return None, None
        if DaskClient is None or DaskLocalCluster is None:
            self._warn(MSG_DASK_UNAVAILABLE)
            return None, None

        n_workers = int(tcfg.get("n_workers", 0))
        threads_per_worker = int(tcfg.get("threads_per_worker", 1))
        memory_limit = str(tcfg.get("memory_limit", "2GB"))
        wait_for_workers = int(tcfg.get("wait_for_workers", max(n_workers, 1)))
        wait_timeout_s = float(tcfg.get("wait_timeout_s", 30))

        if n_workers <= 0:
            return None, None

        client = None
        cluster = None
        try:
            cluster = DaskLocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                dashboard_address=":8787",
            )
            client = DaskClient(cluster)
            client.wait_for_workers(wait_for_workers, timeout=wait_timeout_s)
            return client, cluster
        except Exception as exc:  # noqa: BLE001
            self._warn(f"{MSG_DASK_FAIL}{exc}")
            try:
                if client is not None:
                    _safe_close(client, self._warn, "DaskClient")
                if cluster is not None:
                    _safe_close(cluster, self._warn, "DaskLocalCluster")
            except Exception as exc2:  # noqa: BLE001
                self._warn(f"{MSG_DASK_CLEAN_FAIL}{exc2}")
            return None, None

    # -------------------------
    # TPOT helpers
    # -------------------------

    def _tpot_trials(self, tcfg: dict[str, Any], client: Any | None, safe_cv: int) -> list[dict[str, Any]]:
        """Génère une liste d’essais de paramètres compatibles TPOT1/TPOT2."""
        scoring_raw = tcfg.get(TPOT_SCORING, DEFAULT_REFIT)
        generations = int(tcfg.get(TPOT_GENERATIONS, 5))
        population_size = int(tcfg.get(TPOT_POP_SIZE, 50))
        verbose2 = int(tcfg.get(TPOT_VERBOSE2, 2))
        verbosity1 = int(tcfg.get(TPOT_VERBOSITY1, 2))
        n_jobs = int(tcfg.get(TPOT_N_JOBS, -1)) if client is None else 1

        trials: list[dict[str, Any]] = []

        # TPOT2 (client + scorers + preprocessing)
        scoring2 = F1_WEIGHTED if isinstance(scoring_raw, str) and scoring_raw.lower() in ("f1", F1_WEIGHTED) else scoring_raw
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
                    "preprocessing": bool(tcfg.get(TPOT_PREPROCESSING, True)),
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

        # TPOT1 legacy (verbosity)
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
        """Instancie TPOTClassifier en essayant plusieurs combinaisons de paramètres tolérantes."""
        if TPOTClassifierType is None:
            return None, ModuleNotFoundError("tpot non installé")
        last_exc: Exception | None = None
        for kw in trials:
            try:
                return TPOTClassifierType(**kw), None  # type: ignore[call-arg]
            except TypeError as exc:
                last_exc = exc
                self._warn(f"{MSG_TPOT_INIT_INCOMPAT}{list(kw.keys())}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._warn(f"{MSG_TPOT_INIT_ERR}{exc}")
                continue
        return None, last_exc

    # -------------------------
    # AutoML runners
    # -------------------------

    def _maybe_run_tpot(self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:  # noqa: C901, PLR0912, PLR0915
        """Exécute TPOT si configuré; sinon None."""
        automl = _as_mapping(spec.get(AUTO_KEY))
        if str(automl.get(AUTO_LIB, "")).lower() != "tpot":
            return None

        tcfg = _as_mapping(automl.get(TPOT_KEY))
        client = None
        cluster = None
        try:
            client, cluster = self._make_dask_client(tcfg)

            requested_cv = int(tcfg.get(TPOT_CV, 5))
            min_per_class = int(pd.Series(y).value_counts().min())
            safe_cv = max(2, min(requested_cv, min_per_class))

            trials = self._tpot_trials(tcfg, client, safe_cv)
            tpot, err = self._instantiate_tpot(trials)
            if tpot is None:
                msg = f"{MSG_TPOT_PARAMS_INCOMPAT}{err}"
                self._warn(msg)
                return {
                    "name": automl.get(AUTO_NAME, "tpot"),
                    "best_score": None,
                    "error": msg,
                    "artifacts": [],
                    "cv_results_path": None,
                }

            t0 = time.time()
            tpot.fit(x, y)  # type: ignore[attr-defined]
            dur = time.time() - t0

            export = bool(tcfg.get(TPOT_EXPORT_BEST, False))
            export_path = self.out_dir / str(tcfg.get(TPOT_EXPORT_PATH, TPOT_DEFAULT_EXPORT))
            artifacts: list[str] = []
            if export:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if hasattr(tpot, "export"):
                        tpot.export(str(export_path))  # type: ignore[attr-defined]
                        artifacts = [str(export_path)]
                    else:
                        pipe: Any = getattr(tpot, "fitted_pipeline_", None)
                        if pipe is not None and joblib_dump is not None:
                            pkl_path = export_path.with_suffix(".pkl") if export_path.suffix == ".py" else export_path
                            joblib_dump(pipe, str(pkl_path))
                            artifacts = [str(pkl_path)]
                        elif _tpot2_apply is not None and joblib_dump is not None:
                            ind: Any = getattr(tpot, "best_individual_", None)
                            if ind is not None:
                                skl_pipe = _tpot2_apply(ind, preprocessing_pipeline=None, export_graphpipeline=False)
                                pkl_path = export_path.with_suffix(".pkl")
                                joblib_dump(skl_pipe, str(pkl_path))
                                artifacts = [str(pkl_path)]
                            else:
                                self._warn(MSG_TPOT_NO_EXPORT)
                        else:
                            self._warn(MSG_TPOT_NO_EXPORT)
                except Exception as exc:  # noqa: BLE001
                    self._warn(f"{MSG_EXPORT_FAIL}{exc}")

            # Score robuste (TPOT1: .score; TPOT2: scorer sur pipeline entraîné)
            scoring_raw = tcfg.get(TPOT_SCORING, DEFAULT_REFIT)
            scoring_name = F1_WEIGHTED if isinstance(scoring_raw, str) and scoring_raw.lower() in ("f1", F1_WEIGHTED) else (str(scoring_raw) if isinstance(scoring_raw, str) else F1_WEIGHTED)
            scorer = _get_scorer_safe(scoring_name)

            if hasattr(tpot, "score"):
                best = float(tpot.score(x, y))  # type: ignore[attr-defined]
            else:
                pipe_for_score: Any = getattr(tpot, "fitted_pipeline_", None)
                if pipe_for_score is None:
                    best = 0.0
                else:
                    try:
                        if scorer is not None:
                            best = float(scorer(pipe_for_score, x, y))
                        else:
                            raise ValueError("no scorer")
                    except Exception:
                        y_pred = pipe_for_score.predict(x)
                        best = _f1_weighted()(y, y_pred)

            return {
                "name": automl.get(AUTO_NAME, "tpot"),
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
                self._warn(f"{MSG_DASK_CLOSE_FAIL}{exc}")

    def _maybe_run_lazy(self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
        """Exécute LazyPredict si configuré; sinon None."""
        automl = _as_mapping(spec.get(AUTO_KEY))
        lib = str(automl.get(AUTO_LIB, "")).lower()
        if lib not in {"lazypredict", "lazy"}:
            return None
        if LazyClassifierType is None:
            msg = MSG_LAZY_UNAVAILABLE
            self._warn(msg)
            return {
                "name": automl.get(AUTO_NAME, "lazypredict"),
                "best_score": None,
                "error": msg,
                "artifacts": [],
                "cv_results_path": None,
            }

        lcfg = _as_mapping(automl.get(LAZY_KEY))
        test_size = float(lcfg.get(LAZY_TEST_SIZE, 0.2))
        x_train, x_test, y_train, y_test = _tts_df(
            x, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        include_raw = lcfg.get("include")
        classifiers = self._resolve_classifiers(include_raw if include_raw else None)

        lazy_kwargs: dict[str, Any] = {
            "verbose": int(lcfg.get(LAZY_VERBOSE, 0)),
            "ignore_warnings": True,
            "custom_metric": self._resolve_metric(lcfg.get("custom_metric")),
        }
        if classifiers:
            lazy_kwargs["classifiers"] = classifiers

        clf = LazyClassifierType(**lazy_kwargs)
        t0 = time.time()
        models_df, _unused = cast(tuple[pd.DataFrame, Any], cast(Any, clf).fit(x_train, x_test, y_train, y_test))
        dur = time.time() - t0

        top_n = int(lcfg.get(LAZY_TOP_N, 25))
        models_df = models_df.head(top_n).copy()

        table_path = self.out_dir / str(lcfg.get(LAZY_TABLE_PATH, LAZY_DEFAULT_CSV))
        if bool(lcfg.get(LAZY_SAVE_TABLE, True)):
            table_path.parent.mkdir(parents=True, exist_ok=True)
            models_df.to_csv(table_path, index=True)

        best_score: float | None = None
        for col in LAZY_SCORE_COLS:
            if col in models_df.columns and len(models_df) > 0:
                try:
                    val: Any = models_df.iloc[0][col]
                    best_score = float(val)
                    break
                except Exception as exc:  # noqa: BLE001
                    self._warn(MSG_SCORE_READ_FAIL.format(col) + f"{exc}")

        return {
            "name": automl.get(AUTO_NAME, "lazypredict"),
            "best_score": best_score,
            "duration_sec": dur,
            "artifacts": [str(table_path)] if bool(lcfg.get(LAZY_SAVE_TABLE, True)) else [],
            "cv_results_path": str(table_path) if bool(lcfg.get(LAZY_SAVE_TABLE, True)) else None,
            "best_params": {},
        }

    # -------------------------
    # DL runner (Keras)
    # -------------------------

    def _maybe_run_dl(self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
        """
        Exécute un pipeline DL si automl.library == 'dl':
        - Parse la config DL, split train/val en ndarray, entraîne, et renvoie summary/métriques/artefacts.
        """
        automl = _as_mapping(spec.get(AUTO_KEY))
        if str(automl.get(AUTO_LIB, "")).lower() != "dl":
            return None

        dl_cfg_raw: dict[str, Any] = _as_mapping(automl.get("dl"))
        dl_cfg = DLConfig(**dl_cfg_raw)

        # Split train/val sur ndarrays (stratify attend un ArrayLike numpy)
        x_values = cast(npt.NDArray[np.float64], cast(Any, x).to_numpy(dtype=np.float64, copy=False))
        y_values = cast(npt.NDArray[np.int64], cast(Any, y).to_numpy(dtype=np.int64, copy=False))
        x_tr, x_val, y_tr, y_val = _tts_np(
            x_values, y_values, test_size=0.2, random_state=self.random_state, stratify=y_values
        )

        out = train_dense(cast(Any, x_tr), cast(Any, y_tr), cast(Any, x_val), cast(Any, y_val), dl_cfg)

        # Sélection d'un score "best_score" compatible (acc / auc si dispo en val_*)
        final: dict[str, Any] = cast(dict[str, Any], out.get("final_metrics", {}) or {})
        best_score = None
        for key in DL_BEST_SCORE_KEYS:
            if key in final:
                try:
                    best_score = float(final[key])
                    break
                except Exception as exc:  # noqa: BLE001
                    self._py_logger.debug("parse metric '%s' failed: %s", key, exc)

        artifacts: list[str] = []
        if out.get("model_path"):
            artifacts.append(cast(str, out["model_path"]))
        if out.get("history_csv"):
            artifacts.append(cast(str, out["history_csv"]))

        return {
            "name": automl.get(AUTO_NAME, "dl_dense"),
            "best_score": best_score,
            "best_params": {},
            "duration_sec": None,
            "cv_results_path": out.get("history_csv"),
            "artifacts": artifacts,
            "summary": out.get("summary"),
            "metrics": final,
        }

    # -------------------------
    # Helpers LazyPredict
    # -------------------------

    @staticmethod
    def _resolve_metric(metric_spec: Any) -> Callable[..., float] | None:
        """Résout un metric custom pour LazyPredict, sinon None."""
        if metric_spec in (None, "", "null"):
            return None
        if callable(metric_spec):
            return cast(Callable[..., float], metric_spec)
        if isinstance(metric_spec, str) and metric_spec.lower().startswith("f1"):
            return _f1_weighted()
        return None

    @staticmethod
    def _resolve_sklearn_class(short_name: str) -> type | None:
        """Résout une classe sklearn via recherche dans modules fréquents."""
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
        """Résout une classe sklearn depuis un nom court ou chemin complet module.Classe."""
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
    def _resolve_classifiers(items: list[Any] | None) -> list[type] | None:
        """Résout une liste de classes sklearn depuis une liste d'items hétérogènes."""
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
        """
        Évalue le pipeline: tente DL, puis TPOT, puis Lazy, sinon sklearn+CV avec export CSV.
        - Retourne un dict de résultats harmonisé incluant artefacts et meilleurs scores.
        """
        out = self._maybe_run_dl(spec, x, y)
        if out is not None:
            return out

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
        cv_type = str(cv_cfg.get(CV_TYPE, SEARCH_GRID))
        search = self._search(cv_type, pipe, scoring, refit, cv, grid, dists, cv_cfg)

        t0 = time.time()
        search.fit(x, y)
        duration = time.time() - t0

        cv_path = self.out_dir / f"{FILE_PREFIX}{spec.get('name', DEFAULT_PIPELINE_NAME)}{FILE_EXT}"
        pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)

        return {
            "name": spec.get("name", DEFAULT_PIPELINE_NAME),
            "best_score": float(search.best_score_),
            "best_params": dict(search.best_params_),
            "duration_sec": duration,
            "cv_results_path": str(cv_path),
            "artifacts": [str(cv_path)],
        }
