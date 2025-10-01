"""structlog-based logger manager bridging to stdlib via ProcessorFormatter.

This manager reads declarative handlers/root_handlers from settings (YAML-driven),
builds logging.config.dictConfig accordingly, and ensures both console and
file handlers (RotatingFileHandler) can coexist with distinct renderers.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

from .logger_manager import LoggerBaseConfig, LoggerManager


class StructlogLoggerManager(LoggerManager):
    """Logger manager integrating structlog with stdlib dictConfig."""

    FORMATTER_CONSOLE = "struct_console"
    FORMATTER_FILE = "struct_file"

    def __init__(self, cfg: LoggerBaseConfig) -> None:
        super().__init__(cfg)
        self._slog = None  # lazy import guard

    def _ensure_parent(self, filename: str | None) -> None:
        if not filename:
            return
        try:
            Path(filename).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover
            pass

    def _build_formatters(self, structlog, json_mode: bool) -> dict[str, Any]:
        timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
        console_renderer = structlog.processors.JSONRenderer() if json_mode else structlog.dev.ConsoleRenderer()
        file_renderer = structlog.dev.ConsoleRenderer(colors=False)
        return {
            self.FORMATTER_CONSOLE: {
                "()": "structlog.stdlib.ProcessorFormatter",
                "processor": console_renderer,
                "foreign_pre_chain": [
                    structlog.processors.add_log_level,
                    timestamper,
                ],
            },
            self.FORMATTER_FILE: {
                "()": "structlog.stdlib.ProcessorFormatter",
                "processor": file_renderer,
                "foreign_pre_chain": [
                    structlog.processors.add_log_level,
                    timestamper,
                ],
            },
        }

    def _handler_from_spec(
        self, name: str, spec: dict[str, Any], defaults: LoggerBaseConfig
    ) -> tuple[str, dict[str, Any], str | None]:
        htype = (spec.get("type") or "console").lower()
        level = spec.get("level") or defaults.level
        formatter_key = spec.get("formatter") or (
            self.FORMATTER_CONSOLE if htype == "console" else self.FORMATTER_FILE
        )

        if htype == "console":
            handler = {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": level,
                "formatter": formatter_key,
            }
            return name, handler, None

        if htype == "rotating_file":
            filename = spec.get("filename") or defaults.file_path
            max_bytes = spec.get("maxBytes") or defaults.file_max_bytes
            backup_count = spec.get("backupCount") or defaults.file_backup_count
            handler = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": filename,
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "encoding": "utf-8",
                "level": level,
                "formatter": formatter_key,
            }
            return name, handler, filename

        # Future: watched_file, syslog, http...
        handler = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "level": level,
            "formatter": formatter_key,
        }
        return name, handler, None

    def configure(self) -> None:
        if self._configured:
            return
        try:
            import structlog

            # structlog → stdlib integration
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso", utc=True),
                    structlog.processors.format_exc_info,
                    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
                ],
                wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, self.cfg.level)),
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            formatters = self._build_formatters(structlog, json_mode=self.cfg.json_mode)

            # Handlers from settings (YAML-driven)
            handlers_cfg = getattr(self.cfg, "handlers", None) or []
            handlers: dict[str, dict[str, Any]] = {}
            root_handlers: list[str] = []
            filenames_to_prepare: list[str] = []

            if not handlers_cfg:
                # Default: console + file (if provided)
                h_name, h_dict, _ = self._handler_from_spec(
                    "console", {"type": "console", "level": self.cfg.level}, self.cfg
                )
                handlers[h_name] = h_dict
                root_handlers.append(h_name)

                if self.cfg.file_path:
                    spec_f = {
                        "type": "rotating_file",
                        "level": self.cfg.level,
                        "filename": self.cfg.file_path,
                        "maxBytes": self.cfg.file_max_bytes,
                        "backupCount": self.cfg.file_backup_count,
                        "formatter": self.FORMATTER_FILE,
                    }
                    h_name_f, h_dict_f, fname = self._handler_from_spec("file", spec_f, self.cfg)
                    handlers[h_name_f] = h_dict_f
                    root_handlers.append(h_name_f)
                    if fname:
                        filenames_to_prepare.append(fname)
            else:
                for idx, spec in enumerate(handlers_cfg):
                    name = spec.get("name") or f"h{idx}"
                    h_name, h_dict, fname = self._handler_from_spec(name, spec, self.cfg)
                    handlers[h_name] = h_dict
                    if fname:
                        filenames_to_prepare.append(fname)
                explicit_root = getattr(self.cfg, "root_handlers", None)
                root_handlers = list(explicit_root) if explicit_root else list(handlers.keys())

            for fname in filenames_to_prepare:
                self._ensure_parent(fname)

            loggers_overrides = {}
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

            if self.cfg.default_fields:
                structlog.contextvars.bind_contextvars(**self.cfg.default_fields)

            self._slog = structlog
            self._configured = True
        except ImportError:  # pragma: no cover
            super().configure()

    def get_logger(self, name: str | None = None):
        if not self._configured:
            self.configure()
        if self._slog is None:
            return super().get_logger(name)
        return self._slog.get_logger(name or self.cfg.app_name)
