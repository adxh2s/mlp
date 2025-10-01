"""Stdlib logger manager and base config.

- LoggerBaseConfig now accepts optional handlers/root_handlers to stay schema-compatible
  with advanced YAML routing, even if this stdlib manager does not consume them.
- The stdlib manager continues to configure console + RotatingFileHandler by default,
  preserving previous behavior without regressions.
"""

from __future__ import annotations

import json
import logging
import logging.config
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _json_default(o: Any) -> Any:
    """Best‑effort JSON serialization for numpy and datetimes."""
    if np is not None:
        if isinstance(o, (np.bool_,)):  # type: ignore[attr-defined]
            return bool(o)
        if isinstance(o, (np.integer,)):  # type: ignore[attr-defined]
            return int(o)
        if isinstance(o, (np.floating,)):  # type: ignore[attr-defined]
            f = float(o)
            return f if math.isfinite(f) else None
        if isinstance(o, (np.ndarray,)):  # type: ignore[attr-defined]
            return o.tolist()
    if isinstance(o, (datetime, date)):
        return (
            o.isoformat()
            if not isinstance(o, datetime)
            else (
                o
                if o.tzinfo
                else datetime.fromtimestamp(o.timestamp(), tz=timezone.utc)
            ).isoformat()
        )
    return o


@dataclass
class LoggerBaseConfig:
    """Typed settings consumed by both stdlib and structlog managers."""

    app_name: str = "mlp"
    level: str = "INFO"
    json_mode: bool = False
    file_path: str | None = None
    file_max_bytes: int = 5 * 1024 * 1024
    file_backup_count: int = 3
    uvicorn_noise_filter: bool = True
    default_fields: dict[str, Any] = field(default_factory=dict)

    # New: accept advanced routing to stay schema‑compatible with LoggerSettings/ConfigManager
    handlers: list[dict[str, Any]] | None = None
    root_handlers: list[str] | None = None


class LoggerManager:
    """Stdlib logging manager: console + rotating file handler by default."""

    FORMATTER_JSON = "json"
    FORMATTER_TEXT = "text"
    HANDLER_CONSOLE = "console"
    HANDLER_FILE = "file"

    def __init__(self, cfg: LoggerBaseConfig) -> None:
        self.cfg = cfg
        self._configured = False

    def _ensure_parent(self) -> None:
        """Create parent directory for file handler if file_path is set."""
        if not self.cfg.file_path:
            return
        try:
            Path(self.cfg.file_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        except Exception:  # pragma: no cover
            pass

    def _build_formatters(self) -> dict[str, dict[str, Any]]:
        """Build formatters mapping for dictConfig."""
        if self.cfg.json_mode:
            return {
                self.FORMATTER_JSON: {
                    "()": "logging.Formatter",
                    "fmt": "%(message)s",
                }
            }
        # Text formatter with level, name, and message
        return {
            self.FORMATTER_TEXT: {
                "()": "logging.Formatter",
                "fmt": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            }
        }

    def _build_handlers(self) -> tuple[dict[str, Any], list[str]]:
        """Build default console + rotating file handlers for stdlib manager."""
        handlers: dict[str, Any] = {}
        root_handlers: list[str] = []

        fmt_key = self.FORMATTER_JSON if self.cfg.json_mode else self.FORMATTER_TEXT

        # Console handler
        handlers[self.HANDLER_CONSOLE] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "level": self.cfg.level,
            "formatter": fmt_key,
        }
        root_handlers.append(self.HANDLER_CONSOLE)

        # File handler (optional)
        if self.cfg.file_path:
            handlers[self.HANDLER_FILE] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": self.cfg.file_path,
                "maxBytes": self.cfg.file_max_bytes,
                "backupCount": self.cfg.file_backup_count,
                "encoding": "utf-8",
                "level": self.cfg.level,
                "formatter": fmt_key,
            }
            root_handlers.append(self.HANDLER_FILE)

        return handlers, root_handlers

    def configure(self) -> None:
        """Configure stdlib logging (idempotent)."""
        if self._configured:
            return

        self._ensure_parent()

        formatters = self._build_formatters()
        handlers, root_handlers = self._build_handlers()

        # Noise filters (optional)
        loggers_overrides: dict[str, Any] = {}
        if self.cfg.uvicorn_noise_filter:
            loggers_overrides.update(
                {
                    "uvicorn": {"level": "WARNING", "propagate": False},
                    "uvicorn.error": {"level": "WARNING", "propagate": False},
                    "uvicorn.access": {"level": "WARNING", "propagate": False},
                }
            )

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": formatters,
                "handlers": handlers,
                "root": {"handlers": root_handlers, "level": self.cfg.level},
                "loggers": loggers_overrides,
            }
        )
        self._configured = True

    def get_logger(self, name: str | None = None):
        """Return a stdlib logger, configuring the system if needed."""
        if not self._configured:
            self.configure()
        return logging.getLogger(name or self.cfg.app_name)
