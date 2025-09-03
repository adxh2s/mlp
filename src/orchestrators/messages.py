from __future__ import annotations

"""
Message orchestrator: localized, structured message emission.

This orchestrator relies on the instrumentation/messages_manager module
for translation (gettext-backed) and on LoggerManager for structured logs.
"""

from pathlib import Path
from typing import Any, Optional

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_manager import MessageManager
from src.instrumentation.messages_taxonomy import (
    MESSAGES_READY,
)

# Constants
LOGGER_NAME = "mlp.orchestrators.messages"
SERVICE_NAME = "messages"
CFG_SECTION = "messages"
CFG_DEFAULT_LOCALE = "fr"
CFG_DEFAULT_LOCALES_DIR = "i18n/locales"
CFG_DEFAULT_DOMAINS = ["general", "eda", "pipelines", "report", "data"]


class _MessagesConfig:
    """Lightweight mapper for the Hydra messages config section."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.enabled: bool = bool(raw.get("enabled", True))
        self.locale: str = str(raw.get("locale", CFG_DEFAULT_LOCALE))
        self.locales_dir: str = str(
            raw.get("locales_dir", CFG_DEFAULT_LOCALES_DIR)
        )
        self.domains: list[str] = list(
            raw.get("domains", list(CFG_DEFAULT_DOMAINS))
        )


class MessageOrchestrator(LoggerMixin):
    """
    Orchestrator for localized messages.

    Provides translation and structured emission helpers for other
    orchestrators, binding common context (service, locale).
    """

    def __init__(
        self,
        cfg_mgr: ConfigManager,
        logger_manager: Optional[LoggerManager] = None,
    ) -> None:
        self.cfg_mgr = cfg_mgr
        raw = cfg_mgr.raw.get("orchestrators", {}).get(CFG_SECTION, {})  # type: ignore[union-attr]
        self.cfg = _MessagesConfig(raw if isinstance(raw, dict) else {})
        self.lm = logger_manager or build_logger_manager(
            cfg_mgr.build_logger_settings()
        )
        self.lm.configure()
        self.LOGGER_NAME = LOGGER_NAME  # for LoggerMixin
        self._init_logger(self.lm)

        locales_root = Path(self.cfg_mgr.project_root) / self.cfg.locales_dir
        self.mm = MessageManager(locales_root, default_locale=self.cfg.locale)

        # Try to bind context if structlog; stdlib logger will ignore bind.
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
        """
        Emit a structured, localized log entry.

        The emitted record contains the technical event key and the
        localized 'msg' plus additional fields for observability.
        """
        text = self.translate(domain, event, **fields)
        payload = {"event": event, "msg": text, "domain": domain, **fields}
        # stdlib path uses extra_fields; structlog path uses bound context
        if hasattr(self.log, level):
            getattr(self.log, level)("event", extra={"extra_fields": payload})

    def run(self) -> dict[str, Any]:
        """
        No-op entrypoint for consistency with other orchestrators.

        It reports discovered .mo availability for configured domains.
        """
        mo_dir = (
            Path(self.cfg_mgr.project_root)
            / self.cfg.locales_dir
            / self.cfg.locale
            / "LC_MESSAGES"
        )
        present = [(d, (mo_dir / f"{d}.mo").exists()) for d in self.cfg.domains]
        self.emit("general", MESSAGES_READY, domains=present)
        return {"domains": present, "locale": self.cfg.locale}
