# src/orchestrators/logger.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.orchestrators.bootstrap import bootstrap_instance
from src.orchestrators.message import MessageOrchestratorApp

LOGGER_DOMAIN = "logger"

DEFAULTS: dict[str, Any] = {
    "backend": "structlog",
    "level": "INFO",
    "json_mode": True,
    "file_path": "logs/mlp.json",  # portable par défaut; Compose peut surcharger via MLP_LOG_FILE
    "file_max_bytes": 10_000_000,
    "file_backup_count": 5,
    "uvicorn_noise_filter": True,
    "app_name": "mlp",
    "default_fields": {},
}


class LoggerOrchestrator:
    """
    Standardise le bootstrap de LoggerManager, crée le parent du fichier si nécessaire,
    applique un override MLP_LOG_FILE si présent, configure le manager, puis émet 'logger_ready'.
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
        self._msg_app = msg_app

    def _prepare_file_path(self, config_manager: ConfigManager, path_value: str | None) -> str | None:
        if not path_value:
            return None

        # Override via env (Docker Compose, CI, etc.)
        env_path = os.getenv("MLP_LOG_FILE")
        effective = Path(env_path) if env_path else Path(path_value)

        # Normalisation relative -> projet
        if not effective.is_absolute():
            base = Path(getattr(config_manager, "project_root", "."))  # ConfigManager expose la racine du projet
            effective = (base / effective).resolve()

        # Création du parent
        effective.parent.mkdir(parents=True, exist_ok=True)
        return str(effective)

    def run(self, config_manager: ConfigManager) -> LoggerManager:
        # 1) Base settings depuis ConfigManager
        settings = config_manager.build_logger_settings()

        # 2) Overrides (bootstrap/context/INI → params)
        for k in (
            "backend",
            "level",
            "json_mode",
            "file_path",
            "file_max_bytes",
            "file_backup_count",
            "uvicorn_noise_filter",
            "app_name",
            "default_fields",
        ):
            v = self._params.get(k)
            if v not in (None, ""):
                setattr(settings, k, v)

        # 3) Normaliser et créer le parent du fichier
        settings.file_path = self._prepare_file_path(config_manager, getattr(settings, "file_path", None))

        # 4) Construction & configuration
        self.lm = build_logger_manager(settings)
        self.lm.configure()

        # 5) Émission de test pour déclencher l'ouverture effective du fichier
        try:
            self.lm.get_logger("bootstrap").info(
                "logger_ready",
                backend=getattr(settings, "backend", None),
                json_mode=getattr(settings, "json_mode", None),
                file=getattr(settings, "file_path", None),
                app=getattr(settings, "app_name", None),
            )
        except Exception:
            # Ne pas bloquer l'application si l'émission échoue
            pass

        # 6) Émission i18n (facultative)
        if self._msg_app is not None:
            self._msg_app.emit(
                LOGGER_DOMAIN,
                "logger_ready",
                backend=getattr(settings, "backend", None),
                json_mode=getattr(settings, "json_mode", None),
            )

        return self.lm
