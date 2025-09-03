from __future__ import annotations

"""
Report orchestrator: render consolidated reports and emit events.
"""

import os
from typing import Any, Dict

from src.config.schemas import AppConfig, ReportConfig
from src.datavisualization.report_renderer import ReportRenderer
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import REPORT_DONE, REPORT_START
from src.orchestrators.messages import MessageOrchestrator

# Constants
REPORTS_DIR = "reports"
TEMPLATES_DIR = "src/templates"
LOGGER_NAME = "mlp.orchestrators.report"
DOMAIN = "report"


class ReportOrchestrator(LoggerMixin):
    """Render consolidated reports from EDA and pipelines outputs."""

    def __init__(
        self,
        cfg: ReportConfig,
        project_dir: str,
        app_cfg: AppConfig,
        lm: LoggerManager,
    ) -> None:
        """
        Initialize report orchestrator.

        Args:
            cfg: Report configuration section.
            project_dir: Project artifacts root.
            app_cfg: Validated application config.
            lm: Logger manager instance.
        """
        self.cfg = cfg
        self.app_cfg = app_cfg
        self.out_dir = os.path.join(project_dir, REPORTS_DIR)
        os.makedirs(self.out_dir, exist_ok=True)

        self.renderer = ReportRenderer(TEMPLATES_DIR)

        self.LOGGER_NAME = LOGGER_NAME
        self._init_logger(lm)

        self.msg: Optional[MessageOrchestrator] = None  # may be injected

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attach a MessageOrchestrator for localized emissions."""
        self.msg = msg

    def run(self, eda_payload: Dict[str, Any], pipe_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Render reports and return artifact metadata."""
        if self.msg:
            self.msg.emit(DOMAIN, REPORT_START, out_dir=self.out_dir)
        else:
            self.log.info(
                "report_start", extra={"extra_fields": {"out_dir": self.out_dir}}
            )

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
            self.log.info(
                "report_done", extra={"extra_fields": {"artifacts": out.get("artifacts")}}
            )

        return out
