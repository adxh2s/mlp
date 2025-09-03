from __future__ import annotations

"""
Data orchestrator: analyze and prepare data for downstream modeling.
"""

from typing import Any

import pandas as pd

from src.config.schemas import DataConfig
from src.instrumentation.data_manager import DataManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.orchestrators.messages import MessageOrchestrator
from src.instrumentation.messages_taxonomy import (
    DATA_INIT,
    DATA_PROCESSING_START,
    DATA_ANALYSIS_COMPLETE,
    DATA_PROCESSING_COMPLETE,
    DATA_ANALYSIS_FAILED,
    DATA_PROCESSING_FAILED,
)

# Constants
LOGGER_NAME = "mlp.orchestrators.data"
DOMAIN = "data"

KEY_SHAPE = "shape"
KEY_COLUMNS = "columns"
KEY_TYPES = "types"
KEY_TARGET_FOUND = "target_found"
KEY_MISSING_VALUES = "missing_values"


class DataOrchestrator(LoggerMixin):
    """
    Orchestrate data preparation workflows using DataManager.

    Coordinates data loading, analysis, cleaning, and preparation tasks.
    """

    def __init__(self, cfg: DataConfig, logger_manager=None) -> None:
        """Initialize DataOrchestrator with configuration."""
        self.cfg = cfg
        self.lm = logger_manager

        self.LOGGER_NAME = LOGGER_NAME
        if self.lm is not None:
            self._init_logger(self.lm)
        else:
            import logging

            self.log = logging.getLogger(LOGGER_NAME)

        self.data_manager = DataManager(
            self.cfg.model_dump() if hasattr(self.cfg, "model_dump") else self.cfg
        )

        self.msg: MessageOrchestrator | None = None  # may be injected

        if self.msg:
            self.msg.emit(DOMAIN, DATA_INIT)
        else:
            self.log.info("data_orchestrator_init")

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attach a MessageOrchestrator for localized emissions."""
        self.msg = msg

    def analyze_raw_data(self, raw_data: Any) -> dict[str, Any]:
        """Analyze raw data and return metadata."""
        try:
            df = self.data_manager.load_from_raw(raw_data)
            analysis = {
                KEY_SHAPE: df.shape,
                KEY_COLUMNS: list(df.columns),
                KEY_TYPES: self.data_manager.infer_column_types(df),
                KEY_MISSING_VALUES: df.isnull().sum().to_dict(),
            }
            target_col = self.data_manager.infer_target_column(df)
            analysis[KEY_TARGET_FOUND] = target_col is not None
            return analysis
        except Exception as exc:  # noqa: BLE001
            if self.msg:
                self.msg.emit(DOMAIN, DATA_ANALYSIS_FAILED, level="error", error=str(exc))
            else:
                self.log.error(
                    "data_analysis_failed",
                    extra={"extra_fields": {"error": str(exc)}},
                )
            raise

    def process_data(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]:
        """Process raw data through the full preparation pipeline."""
        if self.msg:
            self.msg.emit(DOMAIN, DATA_PROCESSING_START)
        else:
            self.log.info("data_processing_start")

        try:
            analysis = self.analyze_raw_data(raw_data)
            if self.msg:
                self.msg.emit(DOMAIN, DATA_ANALYSIS_COMPLETE, **analysis)
            else:
                self.log.info(
                    "data_analysis_complete", extra={"extra_fields": analysis}
                )

            X, y = self.data_manager.prepare_for_ml(raw_data)

            result_meta = {
                "features_shape": X.shape,
                "target_shape": y.shape if y is not None else None,
                "feature_columns": list(X.columns),
                "has_target": y is not None,
            }

            if self.msg:
                self.msg.emit(DOMAIN, DATA_PROCESSING_COMPLETE, **result_meta)
            else:
                self.log.info(
                    "data_processing_complete", extra={"extra_fields": result_meta}
                )

            return X, y
        except Exception as exc:  # noqa: BLE001
            if self.msg:
                self.msg.emit(DOMAIN, DATA_PROCESSING_FAILED, level="error", error=str(exc))
            else:
                self.log.error(
                    "data_processing_failed",
                    extra={"extra_fields": {"error": str(exc)}},
                )
            raise

    def run(self, raw_data: Any) -> dict[str, Any]:
        """Main orchestration method: analyze and process data."""
        X, y = self.process_data(raw_data)
        return {
            "X": X,
            "y": y,
            "metadata": {
                "features_count": X.shape[1],
                "samples_count": X.shape,
                "has_target": y is not None,
                "target_classes": len(y.unique()) if y is not None else None,
            },
        }
