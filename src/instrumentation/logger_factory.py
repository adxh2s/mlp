"""Factory for building logging managers (stdlib or structlog) from settings."""

from __future__ import annotations

from ..config.schemas import LoggerSettings
from .config_manager import ConfigManager
from .logger_manager import LoggerBaseConfig, LoggerManager
from .logger_manager_structlog import StructlogLoggerManager


def build_logger_manager(settings: LoggerSettings) -> LoggerManager:
    """Build a logger manager instance based on the configured backend."""
    cfg = LoggerBaseConfig(
        app_name=settings.app_name,
        level=settings.level,
        json_mode=settings.json_mode,
        file_path=settings.file_path,
        file_max_bytes=settings.file_max_bytes,
        file_backup_count=settings.file_backup_count,
        uvicorn_noise_filter=settings.uvicorn_noise_filter,
        default_fields=settings.default_fields,
        handlers=getattr(settings, "handlers", None),
        root_handlers=getattr(settings, "root_handlers", None),
    )
    if settings.backend.lower() == "structlog":
        return StructlogLoggerManager(cfg)
    return LoggerManager(cfg)


def build_logger_manager_from_config(cfg_mgr: ConfigManager) -> LoggerManager:
    """Shortcut: build a logger manager from ConfigManager-derived LoggerSettings."""
    settings = cfg_mgr.build_logger_settings()
    return build_logger_manager(settings)
