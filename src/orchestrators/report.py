from __future__ import annotations

import os
from typing import Any, cast

from src.config.schemas import AppConfig, ReportConfig
from src.datavisualization.report_renderer import ReportRenderer
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.message_taxonomy import REPORT_DONE, REPORT_START
from src.orchestrators.message import MessageOrchestratorApp  # alignement app-level

REPORT_DIR = "report"
TEMPLATES_DIR = "src/templates"
LOGGER_NAME = "mlp.orchestrators.report"
DOMAIN = "report"


class ReportOrchestrator(LoggerMixin):
    """Render consolidated report from EDA and pipeline outputs."""

    def __init__(
        self,
        cfg: ReportConfig,
        project_dir: str,
        app_cfg: AppConfig,
        logger_manager: LoggerManager,
        ctx: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.app_cfg = app_cfg
        self.ctx = ctx or {}

        if self.ctx.get("report_dir"):
            self.out_dir = self.ctx["report_dir"]
        else:
            self.out_dir = os.path.join(project_dir, REPORT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)

        self.renderer = ReportRenderer(TEMPLATES_DIR)

        self.LOGGER_NAME = LOGGER_NAME
        self._init_logger(cast(Any, logger_manager))
        self.log: Any = getattr(self, "log", None)

        self.msg: MessageOrchestratorApp | None = None

    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        self.msg = msg

    def run(self, eda_payload: dict[str, Any], pipe_payload: dict[str, Any]) -> dict[str, Any]:
        if self.msg:
            self.msg.emit(DOMAIN, REPORT_START, out_dir=self.out_dir)
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("report_start", extra={"extra_fields": {"out_dir": self.out_dir}})

        out = self.renderer.render(
            out_dir=self.out_dir,
            project_name=self.app_cfg.project.name,
            formats=self.cfg.formats,
            eda_payload=eda_payload,
            pipe_payload=pipe_payload,
        )

        if self.msg:
            self.msg.emit(DOMAIN, REPORT_DONE, artifacts=out.get("artifacts"))
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("report_done", extra={"extra_fields": {"artifacts": out.get("artifacts")}})

        return out
