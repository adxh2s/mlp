from __future__ import annotations

from typing import Final

"""
Taxonomie des clés de message (gettext msgid) utilisées dans l’application.
"""

APP_START: Final[str] = "app_start"
APP_DONE: Final[str] = "app_done"
STREAMLIT_INIT: Final[str] = "streamlit_init"

GENERAL_START_FROM_DATA: Final[str] = "general_start_from_data"
GENERAL_START_FROM_FILES: Final[str] = "general_start_from_files"
GENERAL_DONE: Final[str] = "general_done"

CONFIG_READY: Final[str] = "config_ready"
CONFIG_ERROR: Final[str] = "config_error"

DATA_INIT: Final[str] = "data_init"
DATA_PROCESSING_START: Final[str] = "data_processing_start"
DATA_ANALYSIS_COMPLETE: Final[str] = "data_analysis_complete"
DATA_PROCESSING_COMPLETE: Final[str] = "data_processing_complete"
DATA_ANALYSIS_FAILED: Final[str] = "data_analysis_failed"
DATA_PROCESSING_FAILED: Final[str] = "data_processing_failed"
DATA_ORCHESTRATOR_DISABLED_NOT_DF: Final[str] = "data_ORCHESTRATOR_disabled_not_df"
DATA_ORCHESTRATOR_FAILED: Final[str] = "data_ORCHESTRATOR_failed"

EDA_START: Final[str] = "eda_start"
EDA_DONE: Final[str] = "eda_done"
EDA_ORCHESTRATOR_FAILED: Final[str] = "eda_ORCHESTRATOR_failed"

FILE_INIT: Final[str] = "file_init"
FILE_ORCHESTRATOR_DISABLED_REQUIRED: Final[str] = "file_ORCHESTRATOR_disabled_required"
FILE_ORCHESTRATOR_FAILED: Final[str] = "file_ORCHESTRATOR_failed"

GENERAL_INIT: Final[str] = "general_init"
GENERAL_DATA_PREVIEW: Final[str] = "general_data_preview"

NO_INPUT_FILE: Final[str] = "no_input_file"
NO_INPUT_FILES_FOUND: Final[str] = "no_input_files_found"
INPUT_FOUND: Final[str] = "input_found"
INPUT_PROCESSED: Final[str] = "input_processed"

MESSAGE_READY: Final[str] = "message_ready"

PIPELINE_START: Final[str] = "pipeline_start"
PIPELINE_EVAL_START: Final[str] = "pipeline_eval_start"
PIPELINE_EVAL_DONE: Final[str] = "pipeline_eval_done"
PIPELINE_DONE: Final[str] = "pipeline_done"
PIPELINE_DISABLED: Final[str] = "pipeline_disabled"
PIPELINE_ORCHESTRATOR_FAILED: Final[str] = "pipeline_ORCHESTRATOR_failed"

REPORT_START: Final[str] = "report_start"
REPORT_DONE: Final[str] = "report_done"
REPORT_ORCHESTRATOR_FAILED: Final[str] = "report_orchestrator_failed"

STEP_ERROR: Final[str] = "step_error"
USING_EXAMPLE_DATA: Final[str] = "using_example_data"
USING_EXAMPLE_DATA_BLOCKED: Final[str] = "using_example_data_blocked"
