from __future__ import annotations

APP_START = "app_start"
APP_DONE = "app_done"

STREAMLIT_INIT = "streamlit_init"

GENERAL_START_FROM_DATA = "general_start_from_data"
GENERAL_START_FROM_FILES = "general_start_from_files"
GENERAL_DONE = "general_done"

CONFIG_READY = "config_ready"
CONFIG_ERROR = "config_error"

DATA_INIT = "data_init",
DATA_PROCESSING_START = "data_processing_start",
DATA_ANALYSIS_COMPLETE = "data_analysis_complete",
DATA_PROCESSING_COMPLETE = "data_processing_complete",
DATA_ANALYSIS_FAILED = "data_analysis_failed",
DATA_PROCESSING_FAILED = "data_processing_failed",

EDA_START = "eda_start"
EDA_DONE = "eda_done"

FILE_INIT = "file_init",
NO_INPUT_FILE = "no_input_file",
INPUT_FOUND = "input_found",
INPUT_PROCESSED = "input_processed" ,

MESSAGES_READY = "messages_ready"

PIPELINES_START = "pipelines_start"
PIPELINES_EVAL_START = "pipelines_eval_start"
PIPELINES_EVAL_DONE = "pipelines_eval_done"
PIPELINES_DONE = "pipelines_done"
PIPELINES_DISABLED = "pipelines_disabled"

REPORT_START = "report_start"
REPORT_DONE = "report_done"

STEP_ERROR = "step_error"  # param: step, error
