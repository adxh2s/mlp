from __future__ import annotations

"""
Logger orchestrator: build and configure LoggerManager, and emit 'logger_ready'.

- Wraps existing LoggerManager/Factory.
- Emits a structured event early so all subsequent components inherit the setup.
"""

from typing import Optional

from omegaconf import DictConfig

from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.orchestrators.messages import MessageOrchestrator
from src.instrumentation.config_manager import ConfigManager


LOGGER_DOMAIN = "logger"


class LoggerOrchestrator:
    """Thin wrapper around LoggerManager to standardize logger bootstrap."""

    def __init__(self, hydra_cfg: DictConfig) -> None:
        """Keep a reference to Hydra config to extract logger settings."""
        self.hydra_cfg = hydra_cfg
        self.lm: Optional[LoggerManager] = None

    def run(self, cfg_mgr: ConfigManager) -> LoggerManager:
        """
        Build and configure LoggerManager, then emit 'logger_ready' once.

        Returns:
            LoggerManager: a configured logger manager instance.
        """
        # 1) Build and configure
        settings = self.hydra_cfg.get("logger")
        self.lm = build_logger_manager(settings)
        self.lm.configure()

        # 2) Emit readiness through shared message orchestrator
        msg = MessageOrchestrator(cfg_mgr, logger_manager=self.lm)
        msg.emit(LOGGER_DOMAIN, "logger_ready", backend=settings.get("backend"), json_mode=settings.get("json_mode"))

        return self.lm
