"""LoggerOrchestrator: bootstrap and configure logging manager.

This orchestrator stays agnostic of Hydra by consuming typed settings from
ConfigManager while still leveraging a generic bootstrap chain for
context/INI/defaults. It also ensures file paths are absolute and parents exist.
"""

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
    "file_path": "logs/mlp.log",
    "file_max_bytes": 10_000_000,
    "file_backup_count": 5,
    "uvicorn_noise_filter": True,
    "app_name": "mlp",
    "default_fields": {},
}


class LoggerOrchestrator:
    """Standardize LoggerManager bootstrap and emit an initial 'logger_ready' event."""

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
        """Create an instance via generic bootstrap (context → INI → defaults)."""
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
        """Attach message orchestrator for i18n events (optional)."""
        self._msg_app = msg_app

    def _normalize_path(self, cfg_mgr: ConfigManager, val: str | None) -> str | None:
        """Resolve to absolute path under project root if relative and create parent."""
        if not val:
            return None
        env_path = os.getenv("MLP_LOG_FILE")
        effective = Path(env_path) if env_path else Path(val)
        if not effective.is_absolute():
            base = Path(getattr(cfg_mgr, "project_root", "."))
            effective = (base / effective).resolve()
        effective.parent.mkdir(parents=True, exist_ok=True)
        return str(effective)

    def run(self, config_manager: ConfigManager) -> LoggerManager:
        """Build and configure the logger manager, then emit 'logger_ready'."""
        # 1) Base settings from Hydra (via ConfigManager)
        settings = config_manager.build_logger_settings()

        # 2) Apply bootstrap overrides (context/INI/defaults → params)
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
            "handlers",
            "root_handlers",
        ):
            v = self._params.get(k)
            if v not in (None, ""):
                setattr(settings, k, v)

        # 3) Normalize final file path and ensure parent exists
        settings.file_path = self._normalize_path(config_manager, getattr(settings, "file_path", None))

        # 4) Build and configure manager
        self.lm = build_logger_manager(settings)
        self.lm.configure()

        # 5) First event to force file creation and validate output routing
        try:
            self.lm.get_logger("bootstrap").info(
                "logger_ready",
                backend=getattr(settings, "backend", None),
                json_mode=getattr(settings, "json_mode", None),
                file=getattr(settings, "file_path", None),
                app=getattr(settings, "app_name", None),
            )
        except Exception:  # pragma: no cover
            pass

        # 6) Optional i18n notification
        if self._msg_app is not None:
            self._msg_app.emit(
                LOGGER_DOMAIN,
                "logger_ready",
                backend=getattr(settings, "backend", None),
                json_mode=getattr(settings, "json_mode", None),
            )

        return self.lm
