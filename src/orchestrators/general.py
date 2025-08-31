from __future__ import annotations

import os
import logging
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

    def __init__(self, cfg_mgr: ConfigManager, logger_manager=None) -> None:
        """Initialize with validated configuration manager."""
        self.cfg_mgr = cfg_mgr
        self.cfg: AppConfig = cfg_mgr.model

        # Output directories
        self.out_dir = self.cfg.project.output_dir
        self.project_dir = os.path.join(self.out_dir, self.cfg.project.name)
        os.makedirs(self.project_dir, exist_ok=True)

        # Logger manager: use injected one if provided, else build from config
        if logger_manager is not None:
            self.lm = logger_manager
        else:
            self.lm = build_logger_manager(self.cfg.logger)
            # configure only if not already configured
            self.lm.configure()

        # Initialize self.log uniformly
        try:
            self._init_logger(self.lm)  # sets self.log via LoggerMixin
        except Exception:
            self.log = logging.getLogger("mlp.orchestrator")

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
            pipes = PipelineOrchestrator(
                orchestrators.pipelines,
                project_dir=self.project_dir,
                random_state=self.cfg.project.random_state,
                out_dir=self.out_dir,
                logger_manager=self.lm
            )
            out[self.KEY_PIPELINES] = pipes.run(X, y)

        if orchestrators.report.enabled:
            rep = ReportOrchestrator(orchestrators.report, self.project_dir, self.cfg, self.lm)
            out[self.KEY_REPORT] = rep.run(out.get(self.KEY_EDA, {}), out.get(self.KEY_PIPELINES, {"results": []}))

        self.log.info(
            "general_done",
            extra={"extra_fields": {"report_artifacts": out.get(self.KEY_REPORT, {}).get("artifacts")}},
        )
        return out
