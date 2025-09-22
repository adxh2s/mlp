from __future__ import annotations

from enum import StrEnum
from typing import Final

# Logger
LOGGER_NAME: Final = "mlp.modeling.pipelines.evaluator"

# CV keys
CV_KEY: Final = "cv"
CV_TYPE: Final = "type"
CV_SCORING: Final = "scoring"
CV_REFIT: Final = "refit"
CV_N_SPLITS: Final = "n_splits"
CV_CV_FOLDS: Final = "cv_folds"
CV_SHUFFLE: Final = "shuffle"
CV_RANDOM_STATE: Final = "random_state"
CV_N_JOBS: Final = "n_jobs"
CV_VERBOSE: Final = "verbose"
CV_RETURN_TRAIN_SCORE: Final = "return_train_score"
CV_ERROR_SCORE: Final = "error_score"
CV_N_ITER: Final = "n_iter"
CV_FACTOR: Final = "factor"

# Search types
SEARCH_GRID: Final = "grid"
SEARCH_RANDOM: Final = "random"
SEARCH_HALVING_GRID: Final = "halving_grid"
SEARCH_HALVING_RANDOM: Final = "halving_random"

# AutoML
class AutoLib(StrEnum):
    TPOT = "tpot"
    LAZY = "lazypredict"
    LAZY_ALT = "lazy"
    DL = "dl"

AUTO_KEY: Final = "automl"
AUTO_LIB: Final = "library"
AUTO_NAME: Final = "name"

# TPOT
TPOT_KEY: Final = "tpot"
TPOT_GENERATIONS: Final = "generations"
TPOT_POP_SIZE: Final = "population_size"
TPOT_SCORING: Final = "scoring"
TPOT_CV: Final = "cv"
TPOT_N_JOBS: Final = "n_jobs"
TPOT_EXPORT_BEST: Final = "export_best_pipeline"
TPOT_EXPORT_PATH: Final = "export_path"
TPOT_DEFAULT_EXPORT: Final = "tpot_best_pipeline.py"
TPOT_VERBOSE2: Final = "verbose"
TPOT_VERBOSITY1: Final = "verbosity"
TPOT_PREPROCESSING: Final = "preprocessing"
F1_WEIGHTED: Final = "f1_weighted"

# LazyPredict
LAZY_KEY: Final = "lazy"
LAZY_TEST_SIZE: Final = "test_size"
LAZY_VERBOSE: Final = "verbose"
LAZY_TOP_N: Final = "top_n"
LAZY_TABLE_PATH: Final = "table_path"
LAZY_SAVE_TABLE: Final = "save_table"
LAZY_DEFAULT_CSV: Final = "lazy_results.csv"
LAZY_SCORE_COLS: Final = ("F1 Score", "Accuracy", "ROC AUC", "Balanced Accuracy")

# Sorties/paths
FILE_PREFIX: Final = "cv_"
FILE_EXT: Final = ".csv"
DEFAULT_REFIT: Final = "f1"
DEFAULT_PIPELINE_NAME: Final = "pipeline"

# Messages
MSG_UNSUPPORTED_CV: Final = "Unsupported cv.type="
MSG_DASK_UNAVAILABLE: Final = "Dask indisponible; exécution TPOT sans client."
MSG_DASK_FAIL: Final = "Échec Dask LocalCluster/Client: "
MSG_DASK_CLEAN_FAIL: Final = "Nettoyage Dask a échoué: "
MSG_TPOT_INIT_INCOMPAT: Final = "TPOT init incompatible avec "
MSG_TPOT_INIT_ERR: Final = "TPOT init erreur: "
MSG_TPOT_PARAMS_INCOMPAT: Final = "Incompatibilité TPOT paramètres: "
MSG_LAZY_UNAVAILABLE: Final = "LazyClassifier non disponible"
MSG_SCORE_READ_FAIL: Final = "Lecture score '{}' impossible: "
MSG_DASK_CLOSE_FAIL: Final = "Fermeture Dask a échoué: "
MSG_EXPORT_FAIL: Final = "Export TPOT a échoué: "
MSG_TPOT_NO_EXPORT: Final = "TPOT sans export ni fitted_pipeline_; aucun artefact exporté."

# DL: clés de métriques candidates utilisées pour best_score
DL_BEST_SCORE_KEYS: Final = (
    "val_accuracy",
    "val_sparse_categorical_accuracy",
    "val_auc",
    "val_AUC",
    "accuracy",
)
