from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_manager import MessageManager
from src.instrumentation.messages_taxonomy import MESSAGES_READY

"""
Message orchestrator: localized, structured message emission.
Relies on messages_manager for translation and LoggerManager for structured logs.
"""

# Constants
LOGGER_NAME = "mlp.orchestrators.messages"
SERVICE_NAME = "messages"
CFG_SECTION = "messages"
CFG_DEFAULT_LOCALE = "fr"
CFG_DEFAULT_LOCALES_DIR = "i18n/locales"
CFG_DEFAULT_DOMAINS = ["general", "eda", "pipelines", "report", "data"]


def _as_dict(obj: Any) -> dict[str, Any]:
    """Coerce obj to dict[str, Any] safely."""
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj)
    if isinstance(obj, Mapping):
        try:
            return {str(k): v for k, v in cast(Mapping[Any, Any], obj).items()}
        except Exception:
            return dict[str, Any]()
    return dict[str, Any]()


class _MessagesConfig:
    """Lightweight mapper for the Hydra messages config section."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.enabled: bool = bool(raw.get("enabled", True))
        self.locale: str = str(raw.get("locale", CFG_DEFAULT_LOCALE))
        self.locales_dir: str = str(raw.get("locales_dir", CFG_DEFAULT_LOCALES_DIR))
        self.domains: list[str] = list(raw.get("domains", list(CFG_DEFAULT_DOMAINS)))


class MessageOrchestrator(LoggerMixin):
    """Orchestrator for localized messages and structured emissions."""

    def __init__(self, config_manager: ConfigManager, logger_manager: LoggerManager | None = None) -> None:
        self.config_manager = config_manager

        # Resolve raw config safely to dict[str, Any]
        raw_root = _as_dict(getattr(config_manager, "raw", {}))
        orch = _as_dict(raw_root.get("orchestrators", {}))
        raw_cfg: dict[str, Any] = _as_dict(orch.get(CFG_SECTION, {}))
        self.cfg = _MessagesConfig(raw_cfg)

        self.lm = logger_manager or build_logger_manager(config_manager.build_logger_settings())
        self.lm.configure()
        self.LOGGER_NAME = LOGGER_NAME
        self._init_logger(cast(Any, self.lm))
        self.log: Any = getattr(self, "log", None)

        locales_root = Path(self.config_manager.project_root) / self.cfg.locales_dir
        self.mm = MessageManager(locales_root, default_locale=self.cfg.locale)

        # Try structlog-style binding; fallback to stdlib logger
        try:
            self.log = self.lm.get_logger(LOGGER_NAME).bind(  # type: ignore[attr-defined]
                service=SERVICE_NAME,
                locale=self.cfg.locale,
            )
        except Exception:
            self.log = self.lm.get_logger(LOGGER_NAME)

    def translate(self, domain: str, key: str, **fields: Any) -> str:
        """Return localized text for domain/key using configured locale."""
        return self.mm.msg(domain=domain, key=key, locale=self.cfg.locale, **fields)

    def emit(self, domain: str, event: str, level: str = "info", **fields: Any) -> None:
        """Emit a structured, localized log entry."""
        text = self.translate(domain, event, **fields)
        payload: dict[str, Any] = {"event": event, "msg": text, "domain": domain, **fields}
        lg: Any = self.log
        if hasattr(lg, level):
            getattr(lg, level)("event", extra={"extra_fields": payload})

    def run(self) -> dict[str, Any]:
        """No-op entrypoint; reports .mo domain availability for observability."""
        mo_dir = Path(self.config_manager.project_root) / self.cfg.locales_dir / self.cfg.locale / "LC_MESSAGES"
        present = [(d, (mo_dir / f"{d}.mo").exists()) for d in self.cfg.domains]
        self.emit("general", MESSAGES_READY, domains=present)
        return {"domains": present, "locale": self.cfg.locale}
