from __future__ import annotations

from omegaconf import DictConfig

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.orchestrators.messages import MessagesOrchestrator

"""
Logger orchestrator: build and configure LoggerManager, and emit 'logger_ready'.
"""

LOGGER_DOMAIN = "logger"


class LoggerOrchestrator:
    """Thin wrapper around LoggerManager to standardize logger bootstrap."""

    def __init__(self, hydra_cfg: DictConfig) -> None:
        self.hydra_cfg = hydra_cfg
        self.lm: LoggerManager | None = None

    def run(self, config_manager: ConfigManager) -> LoggerManager:
        """
        Build and configure LoggerManager, then emit 'logger_ready' once.
        Returns a configured LoggerManager instance.
        """
        settings = self.hydra_cfg.get("logger")
        self.lm = build_logger_manager(settings)
        self.lm.configure()

        msg = MessagesOrchestrator(config_manager, logger_manager=self.lm)
        msg.emit(
            LOGGER_DOMAIN,
            "logger_ready",
            backend=settings.get("backend"),
            json_mode=settings.get("json_mode"),
        )
        return self.lm
