from __future__ import annotations

# Standard library
from typing import Any, tuple

# Third-party
import pandas as pd

# Local
from src.config.schemas import DataConfig
from src.instrumentation.data_manager import DataManager
from src.instrumentation.logger_mixin import LoggerMixin


class DataOrchestrator(LoggerMixin):
    """Orchestrate data preparation workflows.

    Coordinates data loading, cleaning, and preparation tasks
    using DataManager for the actual transformations.
    """

    # Class constants
    LOGGER_NAME = "mlp.orchestrators.data"
    KEY_SHAPE = "shape"
    KEY_COLUMNS = "columns"
    KEY_TYPES = "types"
    KEY_TARGET_FOUND = "target_found"
    KEY_MISSING_VALUES = "missing_values"

    def __init__(self, cfg: DataConfig, logger_manager=None) -> None:
        """Initialize DataOrchestrator with configuration."""
        self.cfg = cfg
        self.lm = logger_manager

        if self.lm is not None:
            self._init_logger(self.lm)
        else:
            import logging

            self.log = logging.getLogger(self.LOGGER_NAME)

        self.data_manager = DataManager(
            self.cfg.model_dump() if hasattr(self.cfg, "model_dump") else self.cfg
        )

        self.log.info("data_orchestrator_init")

    def analyze_raw_data(self, raw_data: Any) -> dict[str, Any]:
        """Analyze raw data and return metadata."""
        try:
            df = self.data_manager.load_from_raw(raw_data)

            analysis = {
                self.KEY_SHAPE: df.shape,
                self.KEY_COLUMNS: list(df.columns),
                self.KEY_TYPES: self.data_manager.infer_column_types(df),
                self.KEY_MISSING_VALUES: df.isnull().sum().to_dict(),
            }

            target_col = self.data_manager.infer_target_column(df)
            analysis[self.KEY_TARGET_FOUND] = target_col is not None

            return analysis

        except Exception as exc:
            self.log.error("data_analysis_failed", extra={"extra_fields": {"error": str(exc)}})
            raise

    def process_data(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]:
        """Process raw data through the full preparation pipeline."""
        self.log.info("data_processing_start")

        try:
            # Analyze raw data first
            analysis = self.analyze_raw_data(raw_data)
            self.log.info("data_analysis_complete", extra={"extra_fields": analysis})

            # Prepare for ML
            X, y = self.data_manager.prepare_for_ml(raw_data)

            result_meta = {
                "features_shape": X.shape,
                "target_shape": y.shape if y is not None else None,
                "feature_columns": list(X.columns),
                "has_target": y is not None,
            }

            self.log.info("data_processing_complete", extra={"extra_fields": result_meta})
            return X, y

        except Exception as exc:
            self.log.error("data_processing_failed", extra={"extra_fields": {"error": str(exc)}})
            raise

    def run(self, raw_data: Any) -> dict[str, Any]:
        """Main orchestration method: analyze and process data."""
        # Process the data
        X, y = self.process_data(raw_data)

        return {
            "X": X,
            "y": y,
            "metadata": {
                "features_count": X.shape[1],
                "samples_count": X.shape[0],
                "has_target": y is not None,
                "target_classes": len(y.unique()) if y is not None else None,
            },
        }
