from __future__ import annotations

"""
General orchestrator: coordinates file intake, data processing, EDA,
pipelines, and reporting with localized, structured logging.

This version composes ConfigOrchestrator and MessageOrchestrator,
and injects the shared message emitter into child orchestrators.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.config.schemas import AppConfig
from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import (
    DATA_ORCH_DISABLED_NOT_DF,
    DATA_ORCH_FAILED,
    EDA_ORCH_FAILED,
    FILE_ORCH_DISABLED_REQUIRED,
    FILE_ORCH_FAILED,
    GENERAL_DONE,
    GENERAL_INIT,
    GENERAL_START_FROM_DATA,
    GENERAL_START_FROM_FILES,
    NO_INPUT_FILES_FOUND,
    PIPES_ORCH_FAILED,
    REPORT_ORCH_FAILED,
    USING_EXAMPLE_DATA,
)
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.data import DataOrchestrator
from src.orchestrators.eda import EDAOrchestrator
from src.orchestrators.file import FileOrchestrator
from src.orchestrators.messages import MessageOrchestrator
from src.orchestrators.pipelines import PipelineOrchestrator
from src.orchestrators.report import ReportOrchestrator

# Constants
LOGGER_NAME = "mlp.orchestrators.general"
DOMAIN = "general"

KEY_FILE = "file"
KEY_DATA = "data"
KEY_EDA = "eda"
KEY_PIPELINES = "pipelines"
KEY_REPORT = "report"


class GeneralOrchestrator(LoggerMixin):
    """
    Coordinate end-to-end flow across orchestrators with message localization.

    The orchestrator builds a shared ConfigOrchestrator and MessageOrchestrator,
    then attaches the message emitter into child orchestrators for consistent,
    localized structured logging across the run.
    """

    def __init__(self, cfg_mgr: ConfigManager, logger_manager: Optional[Any] = None) -> None:
        """Initialize with validated configuration manager."""
        self.cfg_mgr = cfg_mgr
        self.cfg: AppConfig = cfg_mgr.model

        self.out_dir = self.cfg.project.output_dir
        self.project_dir = os.path.join(self.out_dir, self.cfg.project.name)
        os.makedirs(self.project_dir, exist_ok=True)

        self.lm = logger_manager or build_logger_manager(self.cfg.logger)
        self.lm.configure()

        self.LOGGER_NAME = LOGGER_NAME
        try:
            self._init_logger(self.lm)
        except Exception:  # noqa: BLE001
            self.log = logging.getLogger(LOGGER_NAME)

        # Shared orchestrators: config + messages
        self.cfg_orch = ConfigOrchestrator(self.cfg_mgr, logger_manager=self.lm)
        self.msg_orch = MessageOrchestrator(self.cfg_mgr, logger_manager=self.lm)

        # Signal initialization
        self.msg_orch.emit(DOMAIN, GENERAL_INIT, project_dir=self.project_dir)

    def load_example_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load an example dataset for development/testing."""
        ds = load_breast_cancer(as_frame=True)
        X = ds.frame.drop(columns=["target"])
        y = ds.frame["target"]
        return X, y

    def _attach_messages(self, *children: Any) -> None:
        """Attach the shared MessageOrchestrator to child orchestrators."""
        for ch in children:
            if hasattr(ch, "attach_messages"):
                ch.attach_messages(self.msg_orch)

    def run_from_files(self) -> dict[str, Any]:
        """Pipeline mode: File -> Data -> EDA -> Pipelines -> Report."""
        results: dict[str, Any] = {}
        orchestrators = self.cfg.orchestrators
        self.msg_orch.emit(DOMAIN, GENERAL_START_FROM_FILES)

        # 1) File orchestrator
        raw_data: Any
        if orchestrators.file and orchestrators.file.enabled:
            try:
                file_orch = FileOrchestrator(orchestrators.file, logger_manager=self.lm)
                self._attach_messages(file_orch)
                file_result = file_orch.process_input()
                results[KEY_FILE] = file_result
                if not file_result.get("found", False):
                    self.msg_orch.emit(DOMAIN, NO_INPUT_FILES_FOUND)
                    return results
                raw_data = file_result["data"]
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, FILE_ORCH_FAILED, level="error", error=str(exc))
                return results
        else:
            self.msg_orch.emit(DOMAIN, FILE_ORCH_DISABLED_REQUIRED, level="error")
            return {}

        # 2) Data orchestrator
        if orchestrators.data.enabled:
            try:
                data_orch = DataOrchestrator(orchestrators.data, logger_manager=self.lm)
                self._attach_messages(data_orch)
                data_result = data_orch.run(raw_data)
                results[KEY_DATA] = data_result
                X, y = data_result["X"], data_result["y"]
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, DATA_ORCH_FAILED, level="error", error=str(exc))
                return results
        else:
            if isinstance(raw_data, pd.DataFrame):
                X, y = raw_data, None
            else:
                self.msg_orch.emit(DOMAIN, DATA_ORCH_DISABLED_NOT_DF, level="error")
                return results

        return self._run_ml_orchestrators(X, y, results)

    def run_from_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> dict[str, Any]:
        """Direct mode: Start with prepared data."""
        results = {
            KEY_DATA: {
                "X": X,
                "y": y,
                "metadata": {
                    "features_count": X.shape[12],
                    "samples_count": X.shape,
                    "has_target": y is not None,
                },
            }
        }
        self.msg_orch.emit(DOMAIN, GENERAL_START_FROM_DATA, shape=str(X.shape))
        return self._run_ml_orchestrators(X, y, results)

    def _run_ml_orchestrators(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute EDA / Pipelines / Report orchestrators."""
        orchestrators = self.cfg.orchestrators

        # EDA
        if orchestrators.eda.enabled:
            try:
                eda = EDAOrchestrator(orchestrators.eda, self.project_dir, self.lm)
                self._attach_messages(eda)
                results[KEY_EDA] = eda.run(X, y)
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, EDA_ORCH_FAILED, level="error", error=str(exc))

        # Pipelines
        if orchestrators.pipelines.enabled and y is not None:
            try:
                pipes = PipelineOrchestrator(
                    orchestrators.pipelines,
                    project_dir=self.project_dir,
                    random_state=self.cfg.project.random_state,
                    logger_manager=self.lm,
                    out_dir=self.out_dir,
                    cfg_mgr=self.cfg_mgr,
                )
                self._attach_messages(pipes)
                results[KEY_PIPELINES] = pipes.run(X, y)
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, PIPES_ORCH_FAILED, level="error", error=str(exc))

        # Report
        if orchestrators.report.enabled:
            try:
                rep = ReportOrchestrator(orchestrators.report, self.project_dir, self.cfg, self.lm)
                self._attach_messages(rep)
                results[KEY_REPORT] = rep.run(
                    results.get(KEY_EDA, {}),
                    results.get(KEY_PIPELINES, {"results": []}),
                )
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, REPORT_ORCH_FAILED, level="error", error=str(exc))

        self.msg_orch.emit(
            DOMAIN,
            GENERAL_DONE,
            report_artifacts=results.get(KEY_REPORT, {}).get("artifacts"),
        )
        return results

    def run(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> dict[str, Any]:
        """
        Run orchestrator - auto-detect mode based on configuration and parameters.

        - File mode if FileOrchestrator enabled and no data passed
        - Direct data mode if X passed
        - Example mode fallback
        """
        if self.cfg.orchestrators.file and self.cfg.orchestrators.file.enabled and X is None and y is None:
            return self.run_from_files()
        if X is not None:
            return self.run_from_data(X, y)
        self.msg_orch.emit(DOMAIN, USING_EXAMPLE_DATA)
        X_example, y_example = self.load_example_data()
        return self.run_from_data(X_example, y_example)
