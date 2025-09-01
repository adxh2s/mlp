from __future__ import annotations

import os
from typing import Any, Dict

from ..config.schemas import AppConfig, ReportConfig
from ..datavisualization.report_renderer import ReportRenderer
from ..instrumentation.logger_manager import LoggerManager  # interface type
from ..instrumentation.logger_mixin import LoggerMixin


class ReportOrchestrator(LoggerMixin):
    """Render consolidated reports from EDA and pipelines outputs."""

    REPORTS_DIR = "reports"
    TEMPLATES_DIR = "src/templates"
    LOGGER_NAME = "mlp.orchestrators.report"

    def __init__(
        self, cfg: ReportConfig, project_dir: str, app_cfg: AppConfig, lm: LoggerManager
    ) -> None:
        """Initialize report orchestrator.

        Args:
            cfg: Report configuration section.
            project_dir: Project artifacts root.
            app_cfg: Validated application config.
            lm: Logger manager instance.
        """
        self.cfg = cfg
        self.app_cfg = app_cfg
        self.out_dir = os.path.join(project_dir, self.REPORTS_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.renderer = ReportRenderer(self.TEMPLATES_DIR)
        self._init_logger(lm)

    def run(self, eda_payload: Dict[str, Any], pipe_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Render reports and return artifact metadata."""
        self.log.info("report_start", extra={"extra_fields": {"out_dir": self.out_dir}})
        out = self.renderer.render(
            out_dir=self.out_dir,
            project_name=self.app_cfg.project.name,
            formats=self.cfg.formats,
            eda_payload=eda_payload,
            pipe_payload=pipe_payload,
        )
        self.log.info("report_done", extra={"extra_fields": {"artifacts": out.get("artifacts")}})
        return out
