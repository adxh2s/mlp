from __future__ import annotations

"""
Config orchestrator: centralize ConfigManager access and project context.

- Loads and validates the AppConfig once.
- Builds LoggerManager and MessageOrchestrator.
- Computes a Hydra-safe project context (ctx) with absolute paths:
  root_dir, outputs_root, project_dir, data_root, data_in, data_out,
  eda_dir, reports_dir.
"""

from pathlib import Path
from typing import Any

from hydra.utils import get_original_cwd

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import CONFIG_ERROR, CONFIG_READY
from src.orchestrators.messages import MessageOrchestrator

# Constants
LOGGER_NAME = "mlp.orchestrators.config"
DOMAIN = "config"


class ConfigOrchestrator(LoggerMixin):
    """Orchestrator responsible for loading config and building project context."""

    def __init__(
        self,
        cfg_mgr: ConfigManager,
        logger_manager: LoggerManager | None = None,
    ) -> None:
        """Initialize with a ConfigManager and optional LoggerManager."""
        self.cfg_mgr = cfg_mgr
        self.LOGGER_NAME = LOGGER_NAME

        self.lm = logger_manager or build_logger_manager(cfg_mgr.build_logger_settings())
        self.lm.configure()
        self._init_logger(self.lm)

        # Shared message orchestrator
        self.msg = MessageOrchestrator(cfg_mgr, logger_manager=self.lm)

        # Load once
        try:
            self.app_cfg = self.cfg_mgr.load()
        except Exception as exc:  # noqa: BLE001
            self.msg.emit(DOMAIN, CONFIG_ERROR, level="error", error=str(exc))
            raise

        # Lazy-initialized context
        self.ctx: dict[str, str] = {}

    def run(self) -> dict[str, str]:
        """Compute and return a Hydra-safe project context (absolute paths)."""
        root = Path(get_original_cwd())

        project_name = self.app_cfg.project.name
        outputs_root = (root / self.app_cfg.project.output_dir).resolve()
        project_dir = (outputs_root / project_name).resolve()

        # File orchestrator roots
        file_cfg = self.app_cfg.orchestrators.file
        data_root = (root / file_cfg.data_dir).resolve()
        data_in = (data_root / file_cfg.in_dir).resolve()
        data_out = (data_root / file_cfg.out_dir).resolve()

        # Project subdirs
        eda_dir = (project_dir / "eda").resolve()
        reports_dir = (project_dir / "reports").resolve()

        # Ensure structure exists
        for d in (project_dir, data_in, data_out, eda_dir, reports_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.ctx = {
            "root_dir": str(root),
            "outputs_root": str(outputs_root),
            "project_dir": str(project_dir),
            "data_root": str(data_root),
            "data_in": str(data_in),
            "data_out": str(data_out),
            "eda_dir": str(eda_dir),
            "reports_dir": str(reports_dir),
        }

        # Signal
        self.msg.emit(
            DOMAIN,
            CONFIG_READY,
            project_name=project_name,
            output_dir=str(outputs_root),
        )
        return self.ctx

    def get_app_config(self) -> Any:
        """Return the validated AppConfig model."""
        return self.app_cfg

    def get_logger_manager(self) -> LoggerManager:
        """Return the configured LoggerManager instance."""
        return self.lm

    def get_config_manager(self) -> ConfigManager:
        """Return the underlying ConfigManager."""
        return self.cfg_mgr

    def get_context(self) -> dict[str, str]:
        """Return the computed project context."""
        return dict(self.ctx)
