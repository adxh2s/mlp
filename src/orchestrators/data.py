from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.schemas import DataConfig
from src.instrumentation.data_manager import DataManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.message_taxonomy import (
    DATA_INIT,
    DATA_PROCESSING_START,
    DATA_ANALYSIS_COMPLETE,
    DATA_PROCESSING_COMPLETE,
    DATA_ANALYSIS_FAILED,
    DATA_PROCESSING_FAILED,
)
from src.orchestrators.bootstrap import bootstrap_instance
from src.orchestrators.message import MessageOrchestratorApp

LOGGER_NAME = "mlp.orchestrators.data"
DOMAIN = "data"

KEY_SHAPE = "shape"
KEY_COLUMNS = "columns"
KEY_TYPES = "types"
KEY_TARGET_FOUND = "target_found"
KEY_MISSING_VALUES = "missing_values"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "encoding": "utf-8",
    "sep": ",",
    "header": 0,
    "target_column": None,
    "na_values": [],
    "infer_datetime": True,
    "limit_rows": None,
    # Optionnel côté DataManager (par défaut True si absent)
    # "auto_detect_target": True,
}


class DataOrchestrator(LoggerMixin):
    """Orchestrate data preparation workflows using DataManager."""

    def __init__(
        self,
        cfg: DataConfig | dict[str, Any],
        logger_manager: LoggerManager | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
    ) -> None:
        # Normaliser en dict pour DataManager
        self.cfg = cfg if isinstance(cfg, dict) else (cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg or {}))
        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager is not None:
            self._init_logger(logger_manager)
        else:
            import logging
            self.log = logging.getLogger(LOGGER_NAME)

        # Délégation système aux méthodes DataManager (IO + prep)
        self.dm = DataManager(self.cfg)
        self.msg: MessageOrchestratorApp | None = message_orchestrator

    @classmethod
    def bootstrap(
        cls,
        *,
        context_provider,
        logger_manager: LoggerManager | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
        ini_filenames: tuple[str, ...] = ("data.ini", "default.ini"),
    ) -> "DataOrchestrator":
        def factory(params: dict[str, Any]) -> "DataOrchestrator":
            return cls(params, logger_manager=logger_manager, message_orchestrator=message_orchestrator)

        def validator(_inst: "DataOrchestrator") -> None:
            return

        return bootstrap_instance(
            name="data",
            factory=factory,
            defaults=DEFAULTS,
            validator=validator,
            context_provider=context_provider,
            ini_filenames=ini_filenames,
        )

    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        self.msg = msg

    def analyze_df(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyse descriptive légère déléguant l'inférence de types au DataManager.
        """
        types = self.dm.infer_column_types(df)
        tcol = self.dm.infer_target_column(df)
        return {
            KEY_SHAPE: df.shape,
            KEY_COLUMNS: list(df.columns),
            KEY_TYPES: types,
            KEY_MISSING_VALUES: df.isnull().sum().to_dict(),
            KEY_TARGET_FOUND: tcol is not None,
        }

    def process_data(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]:
        # Télémetrie de début
        if self.msg:
            self.msg.emit(DOMAIN, DATA_PROCESSING_START)
        else:
            getattr(self, "log", None) and self.log.info("data_processing_start")

        # 1) Chargement via DataManager (pas d'IO direct ici)
        df0 = self.dm.load_from_raw(
            raw_data,
            encoding=self.cfg.get("encoding"),
            sep=self.cfg.get("sep"),
        )

        # 2) Analyse (types, NA, cible trouvée) pour instrumentation
        analysis = self.analyze_df(df0)
        if self.msg:
            self.msg.emit(DOMAIN, DATA_ANALYSIS_COMPLETE, **analysis)
        else:
            getattr(self, "log", None) and self.log.info("data_analysis_complete", extra={"extra_fields": analysis})

        # 3) Préparation ML complète via DataManager: clean → split → validate
        X, y = self.dm.prepare_for_ml(df0)

        # 4) Fin de traitement (metrics shape/features)
        meta = {
            "features_shape": X.shape,
            "target_shape": (None if y is None else y.shape),
            "feature_columns": list(X.columns),
            "has_target": y is not None,
        }
        if self.msg:
            self.msg.emit(DOMAIN, DATA_PROCESSING_COMPLETE, **meta)
        else:
            getattr(self, "log", None) and self.log.info("data_processing_complete", extra={"extra_fields": meta})

        return X, y

    def run(self, raw_data: Any) -> dict[str, Any]:
        try:
            if self.msg:
                self.msg.emit(DOMAIN, DATA_INIT)
            else:
                getattr(self, "log", None) and self.log.info("data_orchestrator_init")

            X, y = self.process_data(raw_data)

            preview_csv = None
            try:
                preview_csv = X.head(5).to_csv(index=False)
            except Exception:
                preview_csv = None

            return {
                "X": X,
                "y": y,
                "metadata": {
                    "features_count": X.shape[1],
                    "samples_count": X.shape[0],
                    "has_target": y is not None,
                    "target_classes": (len(y.unique()) if y is not None else None),
                },
                "preview_csv": preview_csv,
                "columns": list(X.columns),
            }
        except Exception as exc:  # noqa: BLE001
            if self.msg:
                self.msg.emit(DOMAIN, DATA_PROCESSING_FAILED, level="error", error=str(exc))
            else:
                getattr(self, "log", None) and self.log.error("data_processing_failed", extra={"extra_fields": {"error": str(exc)}})
            raise
