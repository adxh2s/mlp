from __future__ import annotations

import logging
import logging.config
from typing import Optional
from pathlib import Path

from .logger_manager import LoggerBaseConfig, LoggerManager


class StructlogLoggerManager(LoggerManager):
    """structlog-based manager inheriting the LoggerManager interface.

    Falls back to stdlib if structlog is not installed.
    """

    FORMATTER_STRUCT = "struct"
    HANDLER_DEFAULT = "default"

    def __init__(self, cfg: LoggerBaseConfig) -> None:
        """Initialize the structlog logger manager."""
        super().__init__(cfg)
        self._slog = None  # lazy import guard

    def configure(self) -> None:
        """Configure structlog and stdlib integration (once)."""
        if self._configured:
            return
        try:
            import structlog

        # Garantir l’existence du dossier parent si un fichier de log est prévu
            if self.cfg.file_path:
                try:
                    Path(self.cfg.file_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                
            timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
            shared = [
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                timestamper,
            ]
            renderer = structlog.processors.JSONRenderer() if self.cfg.json_mode else structlog.dev.ConsoleRenderer()

            structlog.configure(
                processors=shared + [structlog.processors.format_exc_info, renderer],
                wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, self.cfg.level)),
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            logging.config.dictConfig(
                {
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        self.FORMATTER_STRUCT: {
                            "()": structlog.stdlib.ProcessorFormatter,
                            "processor": renderer,
                            "foreign_pre_chain": [
                                structlog.processors.add_log_level,
                                timestamper,
                            ],
                        }
                    },
                    "handlers": {
                        self.HANDLER_DEFAULT: {
                            "class": "logging.StreamHandler",
                            "formatter": self.FORMATTER_STRUCT,
                        }
                    },
                    "root": {"handlers": [self.HANDLER_DEFAULT], "level": self.cfg.level},
                    "loggers": {
                        **(
                            {
                                "uvicorn": {"level": "WARNING", "propagate": False},
                                "uvicorn.error": {"level": "WARNING", "propagate": False},
                                "uvicorn.access": {"level": "WARNING", "propagate": False},
                            }
                            if self.cfg.uvicorn_noise_filter
                            else {}
                        )
                    },
                }
            )
            if self.cfg.default_fields:
                structlog.contextvars.bind_contextvars(**self.cfg.default_fields)
            self._slog = structlog
            self._configured = True
        except ImportError:
            # Fallback: stdlib configuration
            super().configure()

    def get_logger(self, name: Optional[str] = None):
        """Return a structlog logger if available, stdlib otherwise."""
        if not self._configured:
            self.configure()
        if self._slog is None:
            return super().get_logger(name)
        return self._slog.get_logger(name or self.cfg.app_name)
