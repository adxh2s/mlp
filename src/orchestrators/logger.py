from __future__ import annotations

from typing import Any

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.orchestrators.bootstrap import bootstrap_instance
from src.orchestrators.message import MessageOrchestrator, MessageOrchestratorApp

LOGGER_DOMAIN = "logger"

DEFAULTS: dict[str, Any] = {
    "backend": "structlog",
    "level": "INFO",
    "json_mode": True,
    "file_path": "/logs/streamlit_app.log",
    "file_max_bytes": 10_000_000,
    "file_backup_count": 5,
    "uvicorn_noise_filter": True,
    "app_name": "streamlit_app",
    "default_fields": {},
}

class LoggerOrchestrator:
    """
    Standardise le bootstrap de LoggerManager et, optionnellement, émet un événement i18n 'logger_ready'.
    """

    def __init__(self, **params: Any) -> None:
        self._params = params
        self.lm: LoggerManager | None = None
        self._msg_app: MessageOrchestratorApp | None = None

    @classmethod
    def bootstrap(
        cls,
        *,
        context_provider,
        ini_filenames: tuple[str, ...] = ("logger.ini", "default.ini"),
    ) -> "LoggerOrchestrator":
        def factory(params: dict[str, Any]) -> "LoggerOrchestrator":
            return cls(**params)

        def validator(_inst: "LoggerOrchestrator") -> None:
            return

        return bootstrap_instance(
            name=LOGGER_DOMAIN,
            factory=factory,
            defaults=DEFAULTS,
            validator=validator,
            context_provider=context_provider,
            ini_filenames=ini_filenames,
        )

    def attach_message_app(self, msg_app: MessageOrchestratorApp) -> None:
        """
        Permet d'émettre 'logger_ready' via l’infrastructure i18n si souhaité.
        """
        self._msg_app = msg_app

    def run(self, config_manager: ConfigManager) -> LoggerManager:
        """
        Construit et configure le LoggerManager à partir du ConfigManager,
        puis applique les overrides fournis via bootstrap/context/INI.
        """
        # Base settings depuis ConfigManager
        settings = config_manager.build_logger_settings()

        # Overrides paramétriques (priorité bootstrap/context/INI)
        for k in ("backend", "level", "json_mode", "file_path", "file_max_bytes", "file_backup_count", "uvicorn_noise_filter", "app_name", "default_fields"):
            v = self._params.get(k)
            if v not in (None, ""):
                setattr(settings, k, v)

        # Construction & configuration
        self.lm = build_logger_manager(settings)
        self.lm.configure()

        # Émission facultative d’un événement i18n
        if self._msg_app is not None:
            self._msg_app.emit(
                LOGGER_DOMAIN,
                "logger_ready",
                backend=getattr(settings, "backend", None),
                json_mode=getattr(settings, "json_mode", None),
            )

        return self.lm
