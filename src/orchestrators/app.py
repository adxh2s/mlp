from __future__ import annotations

from pathlib import Path
from typing import Any

# Décorateurs: import robuste avec fallback no-op
try:
    from src.instrumentation.decorators import log_call
except Exception:  # pragma: no cover
    from typing import Callable, TypeVar, ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

    def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
        def deco(fn: Callable[P, T]) -> Callable[P, T]:
            return fn
        return deco

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_manager import LoggerManager
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.logger import LoggerOrchestrator
from src.orchestrators.message import MessageOrchestrator, MessageOrchestratorApp  # core + wrapper


class AppOrchestrator:
    """
    Initialise l'application dans l'ordre:
    1) ConfigOrchestrator (Hydra + contexte + arborescences),
    2) MessageOrchestrator (i18n),
    3) LoggerOrchestrator (fichier+format structuré),
    puis journalise des dumps de contrôle (App/Config/Message/Logger).
    """

    @log_call("app.__init__")
    def __init__(self, hydra_cfg: DictConfig) -> None:
        # 1) Config manager (accès à Hydra et paramètres globaux)
        self.config_manager = ConfigManager(hydra_cfg)

        # 2) ConfigOrchestrator (bootstrap + run + validate)
        self.config_orchestrator = ConfigOrchestrator.bootstrap(self.config_manager)
        app_config = self.config_orchestrator.get_app_config()
        self.context: dict[str, str] = self.config_orchestrator.get_context()

        # 3) MessageOrchestrator via bootstrap (i18n disponible tôt)
        core = MessageOrchestrator.bootstrap(context_provider=lambda _name: {})
        self.message_orchestrator = MessageOrchestratorApp(core)

        # 4) LoggerOrchestrator via bootstrap, puis configuration effective
        self.logger_orchestrator = LoggerOrchestrator.bootstrap(context_provider=lambda _name: {})
        self.logger_orchestrator.attach_message_app(self.message_orchestrator)
        self.logger_manager: LoggerManager = self.logger_orchestrator.run(self.config_manager)

        # 5) Journalisation de contrôle (résumé de config + contexte)
        log = self.logger_manager.get_logger("app.boot")

        # App résumé
        try:
            root = Path(get_original_cwd())
        except Exception:
            root = Path(self.config_manager.project_root).resolve()
        log.info(
            "app_boot",
            project_name=getattr(app_config.project, "name", None),
            output_dir=getattr(app_config.project, "output_dir", None),
            root=str(root),
        )

        # Dump orchestrators.enabled (résumé compact Hydra)
        resolved = OmegaConf.to_container(hydra_cfg, resolve=True) if hydra_cfg is not None else {}
        orch = (resolved.get("orchestrators") or {}) if isinstance(resolved, dict) else {}
        log.info(
            "orchestrators_resolved",
            file_enabled=(orch.get("file") or {}).get("enabled"),
            data_enabled=(orch.get("data") or {}).get("enabled"),
            eda_enabled=(orch.get("eda") or {}).get("enabled"),
            pipeline_enabled=(orch.get("pipeline") or {}).get("enabled"),
            report_enabled=(orch.get("report") or {}).get("enabled"),
        )

        # Contexte chemins clés
        log.info(
            "context_paths",
            project_dir=self.context.get("project_dir"),
            outputs_root=self.context.get("outputs_root"),
            data_in=self.context.get("data_in"),
            data_out=self.context.get("data_out"),
            eda_dir=self.context.get("eda_dir"),
            report_dir=self.context.get("report_dir"),
        )

        # État i18n minimal
        log.info(
            "i18n_ready",
            locales_dir=getattr(core, "localedir", None),
            default_lang=getattr(core, "default_lang", None),
        )

        # Logger effectif
        logger_manager_cfg = getattr(self.logger_manager, "cfg", None)
        log.info(
            "logger_config",
            backend=getattr(logger_manager_cfg, "backend", None),
            level=getattr(logger_manager_cfg, "level", None),
            json_mode=getattr(logger_manager_cfg, "json_mode", None),
            file_path=getattr(logger_manager_cfg, "file_path", None),
            app_name=getattr(logger_manager_cfg, "app_name", None),
        )

    @property
    def logger(self):
        return self.logger_manager.get_logger("app")
