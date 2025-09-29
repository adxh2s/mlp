from __future__ import annotations

import os
from typing import Any

from src.config.schemas import AppConfig, ReportConfig
from src.datavisualization.report_renderer import ReportRenderer
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.message_taxonomy import REPORT_DONE, REPORT_START
from src.orchestrators.bootstrap import bootstrap_instance
from src.orchestrators.message import MessageOrchestratorApp

REPORT_DIR = "report"
TEMPLATES_DIR = "src/templates"
LOGGER_NAME = "mlp.orchestrators.report"
DOMAIN = "report"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "templates_dir": TEMPLATES_DIR,
    "out_dir": REPORT_DIR,
    "formats": ["html"],
}

class ReportOrchestrator(LoggerMixin):
    """Render consolidated report from EDA and pipeline outputs."""

    def __init__(
        self,
        cfg: ReportConfig | dict[str, Any],
        project_dir: str,
        app_cfg: AppConfig,
        logger_manager: LoggerManager | None = None,
        ctx: dict[str, str] | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
    ) -> None:
        # Normaliser la config en dict
        self.cfg = cfg if isinstance(cfg, dict) else cfg.model_dump()
        self.app_cfg = app_cfg
        self.ctx = ctx or {}
        self.project_dir = project_dir

        # Résolution out_dir (valeur cfg relative au project_dir ou absolue)
        out_dir_cfg = self.cfg.get("out_dir", REPORT_DIR)
        self.out_dir = os.path.join(project_dir, str(out_dir_cfg)) if not os.path.isabs(out_dir_cfg) else str(out_dir_cfg)
        os.makedirs(self.out_dir, exist_ok=True)

        # Renderer (délégué unique pour le rendu)
        templates_dir = str(self.cfg.get("templates_dir", TEMPLATES_DIR))
        self.renderer = ReportRenderer(templates_dir=templates_dir)

        # Logger / messages
        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager:
            self._init_logger(logger_manager)
        self.msg = message_orchestrator

    @classmethod
    def bootstrap(
        cls,
        *,
        context_provider,
        project_dir: str,
        app_config: AppConfig,
        logger_manager: LoggerManager | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
        ini_filenames: tuple[str, ...] = ("report.ini", "default.ini"),
    ) -> "ReportOrchestrator":
        def factory(params: dict[str, Any]) -> "ReportOrchestrator":
            ctx = params.pop("_ctx", {})
            return cls(
                cfg=params,
                project_dir=project_dir,
                app_cfg=app_config,
                logger_manager=logger_manager,
                ctx=ctx,
                message_orchestrator=message_orchestrator,
            )

        def validator(inst: "ReportOrchestrator") -> None:
            if not inst.cfg.get("enabled", True):
                return
            os.makedirs(inst.out_dir, exist_ok=True)

        def wrapped_context_provider(_name: str) -> dict[str, Any] | None:
            ctx = context_provider("report") or {}
            params = dict(ctx.get("orchestrators", {}).get("report", {})) if isinstance(ctx.get("orchestrators"), dict) else {}
            params["_ctx"] = ctx
            return params

        return bootstrap_instance(
            name="report",
            factory=factory,
            defaults=DEFAULTS,
            validator=validator,
            context_provider=wrapped_context_provider,
            ini_filenames=ini_filenames,
        )

    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        self.msg = msg

    def run(self, eda_payload: dict[str, Any], pipeline_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.cfg.get("enabled", True):
            return {}

        project_name = getattr(self.app_cfg.project, "name", "project")

        # Évènement de début
        if self.msg:
            self.msg.emit(DOMAIN, REPORT_START, project=project_name, out_dir=self.out_dir)
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("report_start", extra={"extra_fields": {"project": project_name, "out_dir": self.out_dir}})

        # Délégation stricte au renderer (API: formats)
        formats = list(self.cfg.get("formats", ["html"]) or ["html"])
        out = self.renderer.render(
            out_dir=self.out_dir,
            project_name=project_name,
            formats=formats,
            eda_payload=eda_payload or {},
            pipe_payload=pipeline_payload or {"results": []},
        )

        # Évènement de fin
        if self.msg:
            self.msg.emit(DOMAIN, REPORT_DONE, artifacts=out.get("artifacts"))
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("report_done", extra={"extra_fields": {"artifacts": out.get("artifacts")}})

        return out
