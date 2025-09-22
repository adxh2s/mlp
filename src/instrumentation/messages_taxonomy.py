from __future__ import annotations

APP_START = "app_start"
APP_DONE = "app_done"

STREAMLIT_INIT = "streamlit_init"

GENERAL_START_FROM_DATA = "general_start_from_data"
GENERAL_START_FROM_FILES = "general_start_from_files"
GENERAL_DONE = "general_done"

CONFIG_READY = "config_ready"
CONFIG_ERROR = "config_error"

DATA_INIT = ("data_init",)
DATA_PROCESSING_START = "data_processing_start"
DATA_ANALYSIS_COMPLETE = "data_analysis_complete"
DATA_PROCESSING_COMPLETE = "data_processing_complete"
DATA_ANALYSIS_FAILED = "data_analysis_failed"
DATA_PROCESSING_FAILED = "data_processing_failed"
DATA_ORCHESTRATOR_DISABLED_NOT_DF = "data_ORCHESTRATOR_disabled_not_df"
DATA_ORCHESTRATOR_FAILED = "data_ORCHESTRATOR_failed"

EDA_START = "eda_start"
EDA_DONE = "eda_done"
EDA_ORCHESTRATOR_FAILED = "eda_ORCHESTRATOR_failed"

FILE_INIT = ("file_init",)
FILE_ORCHESTRATOR_DISABLED_REQUIRED = "file_ORCHESTRATOR_disabled_required"
FILE_ORCHESTRATOR_FAILED = "file_ORCHESTRATOR_failed"

GENERAL_INIT = "general_init"
GENERAL_DATA_PREVIEW = "general_data_preview"

NO_INPUT_FILE = "no_input_file"
NO_INPUT_FILES_FOUND = "no_input_files_found"
INPUT_FOUND = "input_found"
INPUT_PROCESSED = "input_processed"

MESSAGES_READY = "messages_ready"

PIPELINES_START = "pipelines_start"
PIPELINES_EVAL_START = "pipelines_eval_start"
PIPELINES_EVAL_DONE = "pipelines_eval_done"
PIPELINES_DONE = "pipelines_done"
PIPELINES_DISABLED = "pipelines_disabled"
PIPELINES_ORCHESTRATOR_FAILED = "pipelines_ORCHESTRATOR_failed"

REPORT_START = "report_start"
REPORT_DONE = "report_done"
REPORT_ORCHESTRATOR_FAILED = "report_orchestrator_failed"

STEP_ERROR = "step_error"

USING_EXAMPLE_DATA = "using_example_data"
USING_EXAMPLE_DATA_BLOCKED = "using_example_data_blocked"
