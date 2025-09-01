from __future__ import annotations

# Standard library
import logging
import os
from typing import Any

# Third-party
import pandas as pd

# Local
from ..config.schemas import AppConfig
from ..instrumentation.config_manager import ConfigManager
from ..instrumentation.logger_factory import build_logger_manager
from ..instrumentation.logger_mixin import LoggerMixin
from .eda import EDAOrchestrator
from .file import FileOrchestrator
from .pipelines import PipelineOrchestrator
from .report import ReportOrchestrator


class GeneralOrchestrator(LoggerMixin):
    """Coordinate file intake, EDA, pipelines, and reporting.

    Orchestrates the end-to-end flow:
    - Optional file intake via FileOrchestrator (copy/compress/load).
    - EDA orchestration.
    - Pipelines orchestration.
    - Report orchestration.
    """

    # Class constants
    KEY_FILE = "file"
    KEY_EDA = "eda"
    KEY_PIPELINES = "pipelines"
    KEY_REPORT = "report"
    LOGGER_NAME = "mlp.orchestrators.general"

    def __init__(self, cfg_mgr: ConfigManager, logger_manager=None) -> None:
        """Initialize with validated configuration manager.

        Args:
            cfg_mgr: Loaded and validated configuration manager.
            logger_manager: Optional LoggerManager for structured logging.
        """
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
            self.lm.configure()

        # Initialize logger
        try:
            self._init_logger(self.lm)
        except Exception:
            self.log = logging.getLogger(self.LOGGER_NAME)

        self.log.info("general_init", extra={"extra_fields": {"project_dir": self.project_dir}})

    def run(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        """Run File → EDA → Pipelines → Report and return outputs.

        Args:
            X: Feature matrix as pandas DataFrame.
            y: Target vector as pandas Series, or None for unsupervised steps.

        Returns:
            Dictionary of section outputs keyed by class constants.
        """
        out: dict[str, Any] = {}
        orchestrators = self.cfg.orchestrators
        self.log.info("general_start")

        # 0) File orchestrator (if present and enabled)
        file_cfg = getattr(orchestrators, "file", None)
        if file_cfg is not None and file_cfg.enabled:
            try:
                fo = FileOrchestrator(file_cfg, logger_manager=self.lm)
                out[self.KEY_FILE] = fo.process_input()
            except Exception as exc:
                self.log.warning(
                    "file_orchestrator_failed", extra={"extra_fields": {"error": str(exc)}}
                )

        # 1) EDA
        if orchestrators.eda.enabled:
            eda = EDAOrchestrator(orchestrators.eda, self.project_dir, self.lm)
            out[self.KEY_EDA] = eda.run(X, y)

        # 2) Pipelines
        if orchestrators.pipelines.enabled and y is not None:
            pipes = PipelineOrchestrator(
                orchestrators.pipelines,
                project_dir=self.project_dir,
                random_state=self.cfg.project.random_state,
                logger_manager=self.lm,
                out_dir=self.out_dir,  # supported if added in PipelineOrchestrator
            )
            out[self.KEY_PIPELINES] = pipes.run(X, y)

        # 3) Report
        if orchestrators.report.enabled:
            rep = ReportOrchestrator(orchestrators.report, self.project_dir, self.cfg, self.lm)
            out[self.KEY_REPORT] = rep.run(
                out.get(self.KEY_EDA, {}), out.get(self.KEY_PIPELINES, {"results": []})
            )

        self.log.info(
            "general_done",
            extra={"extra_fields": {"report_artifacts": out.get(self.KEY_REPORT, {}).get("artifacts")}},
        )
        return out
