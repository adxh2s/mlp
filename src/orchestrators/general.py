from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

from ..instrumentation.config_manager import ConfigManager
from ..config.schemas import AppConfig
from ..instrumentation.logger_factory import build_logger_manager
from ..instrumentation.logger_mixin import LoggerMixin
from .eda import EDAOrchestrator
from .pipelines import PipelineOrchestrator
from .report import ReportOrchestrator


class GeneralOrchestrator(LoggerMixin):
    """Coordinate EDA, pipeline evaluation, and report rendering."""

    KEY_EDA = "eda"
    KEY_PIPELINES = "pipelines"
    KEY_REPORT = "report"
    LOGGER_NAME = "mlp.orchestrators.general"

    def __init__(self, cfg_mgr: ConfigManager) -> None:
        """Initialize with validated configuration manager."""
        self.cfg_mgr = cfg_mgr
        self.cfg: AppConfig = cfg_mgr.model
        self.project_dir = os.path.join(self.cfg.project.output_dir, self.cfg.project.name)
        os.makedirs(self.project_dir, exist_ok=True)

        # Build logger manager from configuration and initialize self.log
        self.lm = build_logger_manager(self.cfg.logger)
        self.lm.configure()
        self._init_logger(self.lm)
        self.log.info("general_init", extra={"extra_fields": {"project_dir": self.project_dir}})

    def run(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run EDA → Pipelines → Report and return outputs."""
        out: Dict[str, Any] = {}
        orchestrators = self.cfg.orchestrators
        self.log.info("general_start")

        if orchestrators.eda.enabled:
            eda = EDAOrchestrator(orchestrators.eda, self.project_dir, self.lm)
            out[self.KEY_EDA] = eda.run(X, y)

        if orchestrators.pipelines.enabled and y is not None:
            pipes = PipelineOrchestrator(orchestrators.pipelines, self.project_dir, self.cfg.project.random_state, self.lm)
            out[self.KEY_PIPELINES] = pipes.run(X, y)

        if orchestrators.report.enabled:
            rep = ReportOrchestrator(orchestrators.report, self.project_dir, self.cfg, self.lm)
            out[self.KEY_REPORT] = rep.run(out.get(self.KEY_EDA, {}), out.get(self.KEY_PIPELINES, {"results": []}))

        self.log.info(
            "general_done",
            extra={"extra_fields": {"report_artifacts": out.get(self.KEY_REPORT, {}).get("artifacts")}},
        )
        return out
