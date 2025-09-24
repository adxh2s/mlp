from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hydra.utils import get_original_cwd

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import CONFIG_ERROR, CONFIG_READY
from src.orchestrators.messages import MessagesOrchestrator

"""
Config orchestrator: centralize ConfigManager access and project context.
- Loads and validates the AppConfig once.
- Builds LoggerManager and MessagesOrchestrator.
- Computes a Hydra-safe project context (ctx) with absolute paths.
"""

# Constants
LOGGER_NAME = "mlp.orchestrators.config"
DOMAIN = "config"


class ConfigOrchestrator(LoggerMixin):
    """Load config, build logger/messages, and compute project context."""

    def __init__(self, config_manager: ConfigManager, logger_manager: LoggerManager | None = None) -> None:
        self.config_manager = config_manager
        self.LOGGER_NAME = LOGGER_NAME
        self.lm = logger_manager or build_logger_manager(config_manager.build_logger_settings())
        self.lm.configure()
        # cast pour satisfaire le protocole SupportsGetLogger de LoggerMixin
        self._init_logger(cast(Any, self.lm))
        self.msg = MessagesOrchestrator(config_manager, logger_manager=self.lm)

        try:
            self.app_cfg = self.config_manager.load()
        except Exception as exc:  # noqa: BLE001
            self.msg.emit(DOMAIN, CONFIG_ERROR, level="error", error=str(exc))
            raise

        self.ctx: dict[str, str] = {}

    def run(self) -> dict[str, str]:
        """Compute and return a Hydra-safe project context (absolute paths)."""
        root = Path(get_original_cwd())
        project_name = self.app_cfg.project.name
        outputs_root = (root / self.app_cfg.project.output_dir).resolve()
        project_dir = (outputs_root / project_name).resolve()

        file_cfg = self.app_cfg.orchestrators.file
        data_root = (root / file_cfg.data_dir).resolve()
        data_in = (data_root / file_cfg.in_dir).resolve()
        data_out = (data_root / file_cfg.out_dir).resolve()

        eda_dir = (project_dir / "eda").resolve()
        reports_dir = (project_dir / "reports").resolve()

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

        self.msg.emit(DOMAIN, CONFIG_READY, project_name=project_name, output_dir=str(outputs_root))
        return self.ctx

    def get_app_config(self) -> Any:
        """Return the validated AppConfig model."""
        return self.app_cfg

    def get_logger_manager(self) -> LoggerManager:
        """Return the configured LoggerManager instance."""
        return self.lm

    def get_config_manager(self) -> ConfigManager:
        """Return the underlying ConfigManager."""
        return self.config_manager

    def get_context(self) -> dict[str, str]:
        """Return the computed project context."""
        return dict(self.ctx)
