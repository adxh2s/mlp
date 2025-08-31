# src/instrumentation/bootstrap_logging.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
from ..config_manager import ConfigManager
from ..logger_factory import build_logger_manager

def init_logging_from_config(cfg_mgr: ConfigManager):
    """
    Construit et configure le logger manager (stdlib/structlog) à partir de ConfigManager.
    Retourne un logger (racine de l'app) prêt à l'emploi.
    """
    # Construit LoggerSettings enrichi (file_path -> <racine>/logs/app.log si absent)
    logger_settings = cfg_mgr.build_logger_settings()
    # S'assure que le dossier parent existe
    if logger_settings.file_path:
        p = Path(logger_settings.file_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

    lm = build_logger_manager(logger_settings)
    # configure() est lazy dans vos managers; get_logger() déclenchera la config au besoin
    logger = lm.get_logger(logger_settings.app_name)
    # Log de démarrage
    logger.info(
        "Logging initialized",
        extra={"extra_fields": {
            "file_path": logger_settings.file_path,
            "level": logger_settings.level,
            "backend": logger_settings.backend,
        }},
    )
    return lm, logger
