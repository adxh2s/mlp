from __future__ import annotations

import json
import logging
import logging.config
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """JSON formatter for stdlib logging.

    Extra fields can be passed via extra={"extra_fields": {...}}.
    """

    KEY_TIMESTAMP = "timestamp"
    KEY_LEVEL = "level"
    KEY_LOGGER = "logger"
    KEY_MESSAGE = "message"
    KEY_MODULE = "module"
    KEY_FUNC = "func"
    KEY_LINE = "line"
    KEY_PROCESS = "process"
    KEY_THREAD = "thread"
    KEY_EXC_INFO = "exc_info"
    KEY_EXTRA = "extra_fields"

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a LogRecord as JSON."""
        payload: Dict[str, Any] = {
            self.KEY_TIMESTAMP: datetime.utcfromtimestamp(record.created).isoformat() + "Z",
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
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class LoggerBaseConfig:
    """Common logger config loaded from ConfigManager."""

    app_name: str = "mlp"
    level: str = "INFO"
    json_mode: bool = False
    file_path: Optional[str] = None
    file_max_bytes: int = 5 * 1024 * 1024
    file_backup_count: int = 3
    uvicorn_noise_filter: bool = True
    default_fields: Dict[str, Any] = field(default_factory=dict)


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

    def _build_dict_config(self) -> Dict[str, Any]:
        """Build a dictConfig mapping for logging.config.dictConfig."""
        formatters: Dict[str, Any] = {}
        handlers: Dict[str, Any] = {}
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

        loggers_overrides: Dict[str, Any] = {}
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
    def _default_fields_filter(common: Dict[str, Any]) -> logging.Filter:
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

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Return a stdlib logger, configuring on first use."""
        if not self._configured:
            self.configure()
        return logging.getLogger(name if name else self.cfg.app_name)
