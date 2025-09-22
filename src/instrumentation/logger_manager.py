from __future__ import annotations

# src/instrumentation/logger_manager.py
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
except Exception:
    np = None  # type: ignore


def _json_default(o: Any) -> Any:
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
            (
                o
                if isinstance(o, datetime) and o.tzinfo
                else datetime.fromtimestamp(o.timestamp(), tz=timezone.utc)
            ).isoformat()
            if isinstance(o, datetime)
            else o.isoformat()
        )
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    return str(o)


class JsonFormatter(logging.Formatter):
    KEY_TIMESTAMP: str = "timestamp"
    KEY_LEVEL: str = "level"
    KEY_LOGGER: str = "logger"
    KEY_MESSAGE: str = "message"
    KEY_MODULE: str = "module"
    KEY_FUNC: str = "func"
    KEY_LINE: str = "line"
    KEY_PROCESS: str = "process"
    KEY_THREAD: str = "thread"
    KEY_EXC_INFO: str = "exc_info"
    KEY_EXTRA: str = "extra_fields"

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            self.KEY_TIMESTAMP: datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            self.KEY_LEVEL: record.levelname.lower(),
            self.KEY_LOGGER: record.name,
            self.KEY_MESSAGE: record.getMessage(),
            self.KEY_MODULE: record.module,
            self.KEY_FUNC: record.funcName,
            self.KEY_LINE: record.lineno,
            self.KEY_PROCESS: record.process,
            self.KEY_THREAD: record.thread,
        }
        if record.exc_info:
            payload[self.KEY_EXC_INFO] = self.formatException(record.exc_info)

        extra = getattr(record, self.KEY_EXTRA, None)
        if isinstance(extra, dict):
            payload.update(extra)

        return json.dumps(payload, ensure_ascii=False, default=_json_default)


@dataclass
class LoggerBaseConfig:
    """Common logger config loaded from ConfigManager."""

    app_name: str = "mlp"
    level: str = "INFO"
    json_mode: bool = False
    file_path: str | None = None
    file_max_bytes: int = 5 * 1024 * 1024
    file_backup_count: int = 3
    uvicorn_noise_filter: bool = True
    default_fields: dict[str, Any] = field(default_factory=dict)


class LoggerManager:
    """Base logger manager using Python stdlib logging + dictConfig."""

    CONSOLE_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"
    HANDLER_CONSOLE = "console"
    HANDLER_FILE = "file"
    FORMATTER_JSON = "json"
    FORMATTER_CONSOLE = "console"

    def __init__(self, cfg: LoggerBaseConfig) -> None:
        """Initialize the stdlib logger manager."""
        self.cfg = cfg
        self._configured = False

    def _build_dict_config(self) -> dict[str, Any]:
        """Build a dictConfig mapping for logging.config.dictConfig."""
        formatters: dict[str, Any] = {}
        handlers: dict[str, Any] = {}
        root_handlers: list[str] = []

        if self.cfg.json_mode:
            formatters[self.FORMATTER_JSON] = {"()": f"{__name__}.JsonFormatter"}
            console_formatter = self.FORMATTER_JSON
        else:
            formatters[self.FORMATTER_CONSOLE] = {
                "format": self.CONSOLE_FMT,
                "datefmt": self.CONSOLE_DATEFMT,
            }
            console_formatter = self.FORMATTER_CONSOLE

        handlers[self.HANDLER_CONSOLE] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": console_formatter,
        }
        root_handlers.append(self.HANDLER_CONSOLE)

        if self.cfg.file_path:
            # S'assurer que le dossier existe
            try:
                Path(self.cfg.file_path).expanduser().resolve().parent.mkdir(
                    parents=True, exist_ok=True
                )
            except Exception:
                # Si le mkdir échoue, on laisse logger.config lancer l'erreur de handler
                pass

            handlers[self.HANDLER_FILE] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": self.cfg.file_path,
                "maxBytes": self.cfg.file_max_bytes,
                "backupCount": self.cfg.file_backup_count,
                "encoding": "utf-8",
                "formatter": console_formatter,
            }
            root_handlers.append(self.HANDLER_FILE)

        loggers_overrides: dict[str, Any] = {}
        if self.cfg.uvicorn_noise_filter:
            loggers_overrides.update(
                {
                    "uvicorn": {"level": "WARNING", "propagate": False},
                    "uvicorn.error": {"level": "WARNING", "propagate": False},
                    "uvicorn.access": {"level": "WARNING", "propagate": False},
                }
            )

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "root": {"level": self.cfg.level, "handlers": root_handlers},
            "loggers": loggers_overrides,
        }

    def configure(self) -> None:
        """Apply stdlib logging configuration once."""
        if self._configured:
            return
        logging.config.dictConfig(self._build_dict_config())
        self._configured = True
        if self.cfg.default_fields:
            logging.getLogger().addFilter(self._default_fields_filter(self.cfg.default_fields))

    @staticmethod
    def _default_fields_filter(common: dict[str, Any]) -> logging.Filter:
        """Attach default extra fields to all log records."""

        class _DefaultFields(logging.Filter):
            KEY_EXTRA = "extra_fields"

            def filter(self, record: logging.LogRecord) -> bool:
                base = getattr(record, self.KEY_EXTRA, None)
                if isinstance(base, dict):
                    merged = dict(common)
                    merged.update(base)
                    setattr(record, self.KEY_EXTRA, merged)
                else:
                    setattr(record, self.KEY_EXTRA, dict(common))
                return True

        return _DefaultFields()

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """Return a stdlib logger, configuring on first use."""
        if not self._configured:
            self.configure()
        return logging.getLogger(name if name else self.cfg.app_name)
