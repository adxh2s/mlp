from __future__ import annotations

"""
Config orchestrator: centralize ConfigManager access and distribution.

This orchestrator ensures a single validated AppConfig is used across
other orchestrators and exposes helper getters for common sections.
It also emits localized, structured events via MessageOrchestrator.
"""

from typing import Any, Optional

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_factory import build_logger_manager
from src.orchestrators.messages import MessageOrchestrator
from src.instrumentation.messages_taxonomy import CONFIG_READY, CONFIG_ERROR

# Constants
LOGGER_NAME = "mlp.orchestrators.config"
DOMAIN = "config"


class ConfigOrchestrator(LoggerMixin):
    """
    Orchestrator responsible for loading and exposing application config.

    It wraps ConfigManager and provides a pre-configured LoggerManager so
    downstream orchestrators can share the same logging backend. Events are
    emitted through MessageOrchestrator for i18n-ready observability.
    """

    def __init__(
        self,
        cfg_mgr: ConfigManager,
        logger_manager: Optional[LoggerManager] = None,
    ) -> None:
        """Initialize with a ConfigManager and optional LoggerManager."""
        self.cfg_mgr = cfg_mgr
        self.LOGGER_NAME = LOGGER_NAME

        # Build a logger manager if none provided.
        self.lm = logger_manager or build_logger_manager(
            cfg_mgr.build_logger_settings()
        )
        self.lm.configure()
        self._init_logger(self.lm)

        # Shared message orchestrator (reused by other orchestrators).
        self.msg = MessageOrchestrator(cfg_mgr, logger_manager=self.lm)

        # Load and validate the application config model once.
        try:
            self.app_cfg = self.cfg_mgr.load()
        except Exception as exc:  # noqa: BLE001
            # Emit a localized error and re-raise for callers to handle.
            self.msg.emit(DOMAIN, CONFIG_ERROR, level="error", error=str(exc))
            raise

        # Signal readiness.
        self.msg.emit(
            DOMAIN,
            CONFIG_READY,
            project_name=self.app_cfg.project.name,
            output_dir=self.app_cfg.project.output_dir,
        )

    def get_app_config(self) -> Any:
        """Return the validated AppConfig model."""
        return self.app_cfg

    def get_logger_manager(self) -> LoggerManager:
        """Return the configured LoggerManager instance."""
        return self.lm

    def get_config_manager(self) -> ConfigManager:
        """Return the underlying ConfigManager."""
        return self.cfg_mgr

    # Convenience helpers (optional)

    def get_project_name(self) -> str:
        """Return configured project name."""
        return self.app_cfg.project.name

    def get_output_dir(self) -> str:
        """Return configured output directory."""
        return self.app_cfg.project.output_dir
