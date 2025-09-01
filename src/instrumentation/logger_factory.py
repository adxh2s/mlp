# src/instrumentation/logger_factory.py
from __future__ import annotations

from ..config.schemas import LoggerSettings
from .config_manager import ConfigManager  # nouvel import
from .logger_manager import LoggerBaseConfig, LoggerManager
from .logger_manager_structlog import StructlogLoggerManager


def build_logger_manager(settings: LoggerSettings) -> LoggerManager:
    """Build a logger manager (stdlib or structlog) from settings."""
    cfg = LoggerBaseConfig(
        app_name=settings.app_name,
        level=settings.level,
        json_mode=settings.json_mode,
        file_path=settings.file_path,
        file_max_bytes=settings.file_max_bytes,
        file_backup_count=settings.file_backup_count,
        uvicorn_noise_filter=settings.uvicorn_noise_filter,
        default_fields=settings.default_fields,
    )
    if settings.backend.lower() == "structlog":
        return StructlogLoggerManager(cfg)
    return LoggerManager(cfg)


def build_logger_manager_from_config(cfg_mgr: ConfigManager) -> LoggerManager:
    """Shortcut: construit LoggerManager à partir de ConfigManager."""
    ls = cfg_mgr.build_logger_settings()
    return build_logger_manager(ls)
