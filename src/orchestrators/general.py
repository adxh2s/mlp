from __future__ import annotations

from pathlib import Path
from typing import Any, cast

# Décorateurs: import robuste avec fallback no-op
try:
    from src.instrumentation.decorators import log_call
except Exception:  # pragma: no cover
    from typing import Callable, TypeVar, ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

    def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
        def deco(fn: Callable[P, T]) -> Callable[P, T]:
            return fn
        return deco

import pandas as pd

from src.config.schemas import AppConfig
from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.message_taxonomy import (
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
    PIPELINE_ORCHESTRATOR_FAILED,
    REPORT_ORCHESTRATOR_FAILED,
    USING_EXAMPLE_DATA,
)
from src.orchestrators.config import ConfigOrchestrator  # délégation config/ctx
from src.orchestrators.data import DataOrchestrator
from src.orchestrators.eda import EDAOrchestrator
from src.orchestrators.file import FileOrchestrator
from src.orchestrators.message import MessageOrchestrator, MessageOrchestratorApp
from src.orchestrators.pipeline import PipelineOrchestrator
from src.orchestrators.report import ReportOrchestrator

"""
General orchestrator: coordinate file→data→EDA→pipeline→report with localized, structured logging.
- Builds or reuses shared LoggerManager/MessageOrchestrator and consumes a context (ctx).
- If no ctx is provided, delegates config/ctx construction to ConfigOrchestrator.
"""

LOGGER_NAME = "mlp.orchestrators.general"
DOMAIN = "general"

KEY_FILE = "file"
KEY_DATA = "data"
KEY_EDA = "eda"
KEY_PIPELINE = "pipeline"
KEY_REPORT = "report"


def _example_data() -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return cast(pd.DataFrame, X), cast(pd.Series, y)


class GeneralOrchestrator(LoggerMixin):
    """Coordinate the full workflow: file → data → EDA → pipeline → report with structured, localized telemetry."""

    @log_call("general.__init__")
    def __init__(
        self,
        config_manager: ConfigManager,
        logger_manager: Any | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
        ctx: dict[str, str] | None = None,
    ) -> None:
        self.config_manager = config_manager
        self.cfg: AppConfig = config_manager.model

        # Logging
        self.lm = logger_manager or self._fallback_logger()
        self._init_logger(cast(Any, self.lm))
        self.LOGGER_NAME = LOGGER_NAME

        # Message (app-level): core + wrapper
        if message_orchestrator is not None:
            self.msg_orch = message_orchestrator
        else:
            core = MessageOrchestrator.bootstrap(context_provider=lambda _name: {})
            self.msg_orch = MessageOrchestratorApp(core)

        # Contexte: injecté ou délégué à ConfigOrchestrator
        if ctx is None:
            cfg_orch = ConfigOrchestrator(self.config_manager, logger_manager=cast(Any, self.lm))
            self.cfg = cast(AppConfig, cfg_orch.get_app_config())
            self.ctx = cfg_orch.run()
        else:
            self.ctx = ctx

        self.project_dir = self.ctx.get("project_dir", ".")
        self.out_dir = self.ctx.get("outputs_root", ".")
        self.msg_orch.emit(DOMAIN, GENERAL_INIT, project_dir=self.project_dir)

    @log_call("general._fallback_logger")
    def _fallback_logger(self):
        lm = build_logger_manager(self.cfg.logger)
        lm.configure()
        return lm

    @log_call("general.load_example_data")
    def load_example_data(self) -> tuple[pd.DataFrame, pd.Series]:
        return _example_data()

    @log_call("general._attach_message")
    def _attach_message(self, *children: Any) -> None:
        for ch in children:
            if hasattr(ch, "attach_message"):
                ch.attach_message(self.msg_orch)

    @log_call("general.run_from_files")
    def run_from_files(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        orchestrators = self.cfg.orchestrators

        self.msg_orch.emit(DOMAIN, GENERAL_START_FROM_FILES)

        # File
        if orchestrators.file and orchestrators.file.enabled:
            try:
                file_orch = FileOrchestrator(
                    orchestrators.file, logger_manager=cast(Any, self.lm), ctx=self.ctx
                )
                self._attach_message(file_orch)
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
                data_orch = DataOrchestrator(orchestrators.data, logger_manager=cast(Any, self.lm))
                self._attach_message(data_orch)
                data_result = data_orch.run(raw_data)
                results[KEY_DATA] = data_result
                x, y = data_result["X"], data_result["y"]

                # Prévisualisation 5 lignes sous le dossier EDA
                eda_dir = Path(self.ctx.get("eda_dir", Path(self.project_dir) / "eda"))
                eda_dir.mkdir(parents=True, exist_ok=True)
                preview_path = eda_dir / "data_preview_head.csv"
                x.head(5).to_csv(preview_path, index=False)
                results["data_preview_path"] = str(preview_path)
                results["data_columns"] = list(x.columns)
                self.msg_orch.emit(DOMAIN, "data_preview_written", rows=5, cols=len(x.columns))
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, DATA_ORCHESTRATOR_FAILED, level="error", error=str(exc))
                return results
        elif isinstance(raw_data, pd.DataFrame):
            x, y = raw_data, None
        else:
            self.msg_orch.emit(DOMAIN, DATA_ORCHESTRATOR_DISABLED_NOT_DF, level="error")
            return results

        return self._run_ml_orchestrators(x, y, results)

    @log_call("general.run_from_data")
    def run_from_data(self, x: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {
            KEY_DATA: {
                "X": x,
                "y": y,
                "metadata": {
                    "features_count": x.shape[1],
                    "samples_count": x.shape[0],
                    "has_target": y is not None,
                },
            }
        }
        self.msg_orch.emit(DOMAIN, GENERAL_START_FROM_DATA, shape=str(x.shape))
        return self._run_ml_orchestrators(x, y, results)

    @log_call("general._run_ml_orchestrators")
    def _run_ml_orchestrators(
        self, x: pd.DataFrame, y: pd.Series | None, results: dict[str, Any]
    ) -> dict[str, Any]:
        orchestrators = self.cfg.orchestrators

        # EDA
        if orchestrators.eda.enabled:
            try:
                eda = EDAOrchestrator(orchestrators.eda, self.project_dir, logger_manager=cast(Any, self.lm))
                self._attach_message(eda)
                results[KEY_EDA] = eda.run(x, y)
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, EDA_ORCHESTRATOR_FAILED, level="error", error=str(exc))

        # Pipeline
        if orchestrators.pipeline.enabled and y is not None:
            try:
                # Résolution out_dir (compatibilité existante)
                p_out_cfg = getattr(orchestrators.pipeline, "out_dir", None)
                if p_out_cfg:
                    p = Path(p_out_cfg)
                    if p.is_absolute():
                        out_dir = str(p)
                    elif p.parts and p.parts[0] == "outputs":
                        root_dir = Path(self.project_dir).parent.parent
                        out_dir = str(root_dir / p_out_cfg)
                    else:
                        out_dir = str(Path(self.project_dir) / p_out_cfg)
                else:
                    out_dir = str(Path(self.project_dir) / "pipeline_cv")

                pipes = PipelineOrchestrator(
                    orchestrators.pipeline,
                    project_dir=self.project_dir,
                    random_state=self.cfg.project.random_state,
                    logger_manager=cast(Any, self.lm),
                    out_dir=out_dir,
                    ctx=self.ctx,
                )
                self._attach_message(pipes)
                results[KEY_PIPELINE] = pipes.run(x, y)
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, PIPELINE_ORCHESTRATOR_FAILED, level="error", error=str(exc))

        # Report
        if orchestrators.report.enabled:
            try:
                rep = ReportOrchestrator(
                    orchestrators.report, self.project_dir, self.cfg, logger_manager=cast(Any, self.lm), ctx=self.ctx
                )
                self._attach_message(rep)
                results[KEY_REPORT] = rep.run(results.get(KEY_EDA, {}), results.get(KEY_PIPELINE, {"results": []}))
            except Exception as exc:  # noqa: BLE001
                self.msg_orch.emit(DOMAIN, REPORT_ORCHESTRATOR_FAILED, level="error", error=str(exc))

        self.msg_orch.emit(DOMAIN, GENERAL_DONE, report_artifacts=results.get(KEY_REPORT, {}).get("artifacts"))
        return results

    @log_call("general.run")
    def run(self, x: pd.DataFrame | None = None, y: pd.Series | None = None) -> dict[str, Any]:
        file_enabled = bool(self.cfg.orchestrators.file and self.cfg.orchestrators.file.enabled)
        self.msg_orch.emit(
            DOMAIN,
            "branch_decision",
            file_enabled=file_enabled,
            x_present=(x is not None),
            y_present=(y is not None),
        )

        if x is None and file_enabled:
            return self.run_from_files()
        if x is not None:
            return self.run_from_data(x, y)

        if not getattr(self.cfg.project, "allow_example_fallback", False):
            self.msg_orch.emit(
                DOMAIN,
                NO_INPUT_FILES_FOUND,
                level="error",
                reason="No input data found; example fallback disabled",
            )
            return {"error": "no_input_data", "fallback_used": False}

        self.msg_orch.emit(DOMAIN, USING_EXAMPLE_DATA)
        x_ex, y_ex = self.load_example_data()
        return self.run_from_data(x_ex, y_ex)
