from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.config.schemas import AppConfig
from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import (
    DATA_ORCHESTRATOR_DISABLED_NOT_DF,
    DATA_ORCHESTRATOR_FAILED,
    EDA_ORCHESTRATOR_FAILED,
    FILE_ORCHESTRATOR_DISABLED_REQUIRED,
    FILE_ORCHESTRATOR_FAILED,
    GENERAL_DONE,
    GENERAL_INIT,
    GENERAL_START_FROM_DATA,
    GENERAL_START_FROM_FILES,
    NO_INPUT_FILES_FOUND,
    PIPELINES_ORCHESTRATOR_FAILED,
    REPORT_ORCHESTRATOR_FAILED,
    USING_EXAMPLE_DATA,
)
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.data import DataOrchestrator
from src.orchestrators.eda import EDAOrchestrator
from src.orchestrators.file import FileOrchestrator
from src.orchestrators.messages import MessageOrchestrator
from src.orchestrators.pipelines import PipelineOrchestrator
from src.orchestrators.report import ReportOrchestrator

"""
General orchestrator: coordinates file intake, data processing, EDA,
pipelines, and reporting with localized, structured logging.

This version expects LoggerManager, MessageOrchestrator, and ctx to be
injected (from AppOrchestrator), but falls back gracefully if not provided.
"""

LOGGER_NAME = "mlp.orchestrators.general"
DOMAIN = "general"
KEY_FILE = "file"
KEY_DATA = "data"
KEY_EDA = "eda"
KEY_PIPELINES = "pipelines"
KEY_REPORT = "report"


class GeneralOrchestrator(LoggerMixin):
    """Coordinate end-to-end flow across orchestrators with message localization."""

    def __init__(
        self,
        cfg_mgr: ConfigManager,
        logger_manager: Optional[Any] = None,
        message_orchestrator: Optional[MessageOrchestrator] = None,
        ctx: Optional[dict[str, str]] = None,
    ) -> None:
        self.cfg_mgr = cfg_mgr
        self.cfg: AppConfig = cfg_mgr.model

        # Logging
        self.lm = logger_manager or self._fallback_logger()
        self._init_logger(self.lm)
        self.LOGGER_NAME = LOGGER_NAME

        # Messages
        self.msg_orch = message_orchestrator or MessageOrchestrator(self.cfg_mgr, logger_manager=self.lm)

        # Context (fallback: minimal if not provided)
        if ctx is None:
            cfg_orch = ConfigOrchestrator(self.cfg_mgr, logger_manager=self.lm)
            out_dir = cfg_orch.get_output_dir() if hasattr(cfg_orch, "get_output_dir") else "."
            self.ctx = {"outputs_root": out_dir, "project_dir": str(Path(out_dir) / self.cfg.project.name)}
        else:
            self.ctx = ctx

        self.project_dir = self.ctx.get("project_dir", ".")
        self.out_dir = self.ctx.get("outputs_root", ".")

        self.msg_orch.emit(DOMAIN, GENERAL_INIT, project_dir=self.project_dir)

    def _fallback_logger(self):

        lm = build_logger_manager(self.cfg.logger)
        lm.configure()
        return lm

    def load_example_data(self) -> tuple[pd.DataFrame, pd.Series]:
        ds = load_breast_cancer(as_frame=True)
        X = ds.frame.drop(columns=["target"])
        y = ds.frame["target"]
        return X, y

    def _attach_messages(self, *children: Any) -> None:
        for ch in children:
            if hasattr(ch, "attach_messages"):
                ch.attach_messages(self.msg_orch)

    def run_from_files(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        orchestrators = self.cfg.orchestrators
        self.msg_orch.emit(DOMAIN, GENERAL_START_FROM_FILES)

        # File
        if orchestrators.file and orchestrators.file.enabled:
            try:
                file_orch = FileOrchestrator(orchestrators.file, logger_manager=self.lm, ctx=self.ctx)
                self._attach_messages(file_orch)
                file_result = file_orch.process_input()
                results[KEY_FILE] = file_result
                if not file_result.get("found", False):
                    self.msg_orch.emit(DOMAIN, NO_INPUT_FILES_FOUND)
                    return results
                raw_data = file_result["data"]
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, FILE_ORCHESTRATOR_FAILED, level="error", error=str(exc))
                return results
        else:
            self.msg_orch.emit(DOMAIN, FILE_ORCHESTRATOR_DISABLED_REQUIRED, level="error")
            return results

        # Data
        if orchestrators.data.enabled:
            try:
                data_orch = DataOrchestrator(orchestrators.data, logger_manager=self.lm)
                self._attach_messages(data_orch)
                data_result = data_orch.run(raw_data)
                results[KEY_DATA] = data_result
                X, y = data_result["X"], data_result["y"]

                eda_dir = Path(self.ctx.get("eda_dir", Path(self.project_dir) / "eda"))
                eda_dir.mkdir(parents=True, exist_ok=True)
                preview_path = eda_dir / "data_preview_head.csv"
                X.head(5).to_csv(preview_path, index=False)
                results["data_preview_path"] = str(preview_path)
                results["data_columns"] = list(X.columns)
                self.msg_orch.emit(DOMAIN, "data_preview_written", rows=5, cols=len(X.columns))
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, DATA_ORCHESTRATOR_FAILED, level="error", error=str(exc))
                return results
        else:
            if isinstance(raw_data, pd.DataFrame):
                X, y = raw_data, None
            else:
                self.msg_orch.emit(DOMAIN, DATA_ORCHESTRATOR_DISABLED_NOT_DF, level="error")
                return results

        return self._run_ml_orchestrators(X, y, results)

    def run_from_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> dict[str, Any]:
        results = {
            KEY_DATA: {
                "X": X,
                "y": y,
                "metadata": {
                    "features_count": X.shape[1],
                    "samples_count": X.shape[0],
                    "has_target": y is not None,
                },
            }
        }
        self.msg_orch.emit(DOMAIN, GENERAL_START_FROM_DATA, shape=str(X.shape))
        return self._run_ml_orchestrators(X, y, results)

    def _run_ml_orchestrators(self, X: pd.DataFrame, y: Optional[pd.Series], results: dict[str, Any]) -> dict[str, Any]:
        orchestrators = self.cfg.orchestrators

        # EDA
        if orchestrators.eda.enabled:
            try:
                eda = EDAOrchestrator(orchestrators.eda, self.project_dir, self.lm)
                self._attach_messages(eda)
                results[KEY_EDA] = eda.run(X, y)
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, EDA_ORCHESTRATOR_FAILED, level="error", error=str(exc))

        # Pipelines
        if orchestrators.pipelines.enabled and y is not None:
            try:
                # Résolution robuste du répertoire de sortie des pipelines
                p_out_cfg = getattr(orchestrators.pipelines, "out_dir", None)
                if p_out_cfg:
                    p = Path(p_out_cfg)
                    if p.is_absolute():
                        out_dir = str(p)
                    elif p.parts and p.parts[0] == "outputs":
                        # Résoudre depuis la racine du repo (root_dir) si disponible, sinon remonter depuis project_dir
                        root_dir = getattr(self, "root_dir", None) or Path(self.project_dir).parent.parent
                        out_dir = str(Path(root_dir) / p_out_cfg)
                    else:
                        out_dir = str(Path(self.project_dir) / p_out_cfg)
                else:
                    out_dir = str(Path(self.project_dir) / "pipelines_cv")

                pipes = PipelineOrchestrator(
                    orchestrators.pipelines,
                    project_dir=self.project_dir,
                    random_state=self.cfg.project.random_state,
                    logger_manager=self.lm,
                    out_dir=out_dir,
                    cfg_mgr=self.cfg_mgr,
                )

                self._attach_messages(pipes)
                results[KEY_PIPELINES] = pipes.run(X, y)
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, PIPELINES_ORCHESTRATOR_FAILED, level="error", error=str(exc))


        # Report
        if orchestrators.report.enabled:
            try:
                rep = ReportOrchestrator(orchestrators.report, self.project_dir, self.cfg, self.lm)
                self._attach_messages(rep)
                results[KEY_REPORT] = rep.run(results.get(KEY_EDA, {}), results.get(KEY_PIPELINES, {"results": []}))
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, REPORT_ORCHESTRATOR_FAILED, level="error", error=str(exc))

        self.msg_orch.emit(DOMAIN, GENERAL_DONE, report_artifacts=results.get(KEY_REPORT, {}).get("artifacts"))
        return results

    def run(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> dict[str, Any]:
        file_enabled = bool(self.cfg.orchestrators.file and self.cfg.orchestrators.file.enabled)
        self.msg_orch.emit(DOMAIN, "branch_decision", file_enabled=file_enabled, x_present=(X is not None), y_present=(y is not None))

        if X is None and file_enabled:
            return self.run_from_files()
        if X is not None:
            return self.run_from_data(X, y)

        if not getattr(self.cfg.project, "allow_example_fallback", False):
            self.msg_orch.emit(DOMAIN, NO_INPUT_FILES_FOUND, level="error", reason="No input data found; example fallback disabled")
            return {"error": "no_input_data", "fallback_used": False}

        self.msg_orch.emit(DOMAIN, USING_EXAMPLE_DATA)
        X_ex, Y_ex = self.load_example_data()
        return self.run_from_data(X_ex, Y_ex)
