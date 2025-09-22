# src/orchestrators/data.py
from __future__ import annotations

"""
Data orchestrator: analyze and prepare data for downstream modeling.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.schemas import DataConfig
from src.instrumentation.data_manager import DataManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import (
    DATA_ANALYSIS_COMPLETE,
    DATA_INIT,
    DATA_PROCESSING_COMPLETE,
    DATA_PROCESSING_FAILED,
    DATA_PROCESSING_START,
)
from src.orchestrators.messages import MessageOrchestrator

LOGGER_NAME = "mlp.orchestrators.data"
DOMAIN = "data"

KEY_SHAPE = "shape"
KEY_COLUMNS = "columns"
KEY_TYPES = "types"
KEY_TARGET_FOUND = "target_found"
KEY_MISSING_VALUES = "missing_values"


class DataOrchestrator(LoggerMixin):
    """Orchestrate data preparation workflows using DataManager."""

    def __init__(self, cfg: DataConfig, logger_manager=None) -> None:
        self.cfg = cfg
        self.lm = logger_manager
        self.LOGGER_NAME = LOGGER_NAME
        if self.lm is not None:
            self._init_logger(self.lm)
        else:
            import logging

            self.log = logging.getLogger(LOGGER_NAME)

        # Passer la config Pydantic en dict au DataManager
        cfg_dict = (
            self.cfg.model_dump() if hasattr(self.cfg, "model_dump") else dict(self.cfg or {})
        )
        self.data_manager = DataManager(cfg_dict)

        self.msg: MessageOrchestrator | None = None
        if self.msg:
            self.msg.emit(DOMAIN, DATA_INIT)
        else:
            self.log.info("data_orchestrator_init")

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        self.msg = msg

    def analyze_df(self, df: pd.DataFrame) -> dict[str, Any]:
        """Rapport synthétique pour le logging amont, incluant la cible résolue."""
        types = self.data_manager.infer_column_types(df)
        cfg_target = getattr(self.cfg, "target_column", None)
        # Priorité absolue à la config si présente et existante
        if cfg_target and cfg_target in df.columns:
            target_col = cfg_target
        else:
            target_col = self.data_manager.infer_target_column(df)
        return {
            KEY_SHAPE: df.shape,
            KEY_COLUMNS: list(df.columns),
            KEY_TYPES: types,
            KEY_MISSING_VALUES: df.isnull().sum().to_dict(),
            KEY_TARGET_FOUND: target_col is not None,
        }

    @staticmethod
    def _load_df_from_payload(raw_data: Any, encoding: str | None, sep: str | None) -> pd.DataFrame:
        if isinstance(raw_data, pd.DataFrame):
            return raw_data
        if isinstance(raw_data, (str, Path)):
            return DataManager.load_csv(Path(raw_data).resolve(), encoding=encoding, sep=sep)
        if isinstance(raw_data, dict) and "path" in raw_data:
            return DataManager.load_csv(
                Path(raw_data["path"]).resolve(), encoding=encoding, sep=sep
            )
        raise ValueError("raw_data must be a DataFrame, a path, or a dict containing 'path'")

    def process_data(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]:
        """Charge → analyse → prépare (clean/split/validate) → retourne X,y."""
        if self.msg:
            self.msg.emit(DOMAIN, DATA_PROCESSING_START)
        else:
            self.log.info("data_processing_start")

        try:
            # 1) Charger avec encodage/séparateur de la config
            df = self._load_df_from_payload(
                raw_data, getattr(self.cfg, "encoding", None), getattr(self.cfg, "sep", None)
            )

            # 2) Analyse
            analysis = self.analyze_df(df)
            if self.msg:
                self.msg.emit(DOMAIN, DATA_ANALYSIS_COMPLETE, **analysis)
            else:
                self.log.info("data_analysis_complete", extra={"extra_fields": analysis})

            # 3) Préparer (clean + split X/y + validate)
            X, y = self.data_manager.prepare_for_ml(df)

            meta = {
                "features_shape": X.shape,
                "target_shape": y.shape if y is not None else None,
                "feature_columns": list(X.columns),
                "has_target": y is not None,
            }
            if self.msg:
                self.msg.emit(DOMAIN, DATA_PROCESSING_COMPLETE, **meta)
            else:
                self.log.info("data_processing_complete", extra={"extra_fields": meta})

            return X, y
        except Exception as exc:  # noqa: BLE001
            if self.msg:
                self.msg.emit(DOMAIN, DATA_PROCESSING_FAILED, level="error", error=str(exc))
            else:
                self.log.error(
                    "data_processing_failed", extra={"extra_fields": {"error": str(exc)}}
                )
            raise

    def run(self, raw_data: Any) -> dict[str, Any]:
        X, y = self.process_data(raw_data)
        preview_csv = None
        try:
            preview_csv = X.head(5).to_csv(index=False)
        except Exception:
            pass
        return {
            "X": X,
            "y": y,
            "metadata": {
                "features_count": X.shape[1],
                "samples_count": X.shape[0],
                "has_target": y is not None,
                "target_classes": len(y.unique()) if y is not None else None,
            },
            "preview_csv": preview_csv,
            "columns": list(X.columns),
        }
