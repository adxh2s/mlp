from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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

import pandas as pd

from src.config.schemas import PipelineConfig
from src.instrumentation.logger_mixin import LoggerMixin, SupportsGetLogger
from src.instrumentation.message_taxonomy import (
    PIPELINE_DISABLED,
    PIPELINE_DONE,
    PIPELINE_EVAL_DONE,
    PIPELINE_EVAL_START,
    PIPELINE_START,
)
from src.modeling.pipeline.evaluator import PipelineEvaluator
from src.orchestrators.bootstrap import bootstrap_instance
from src.orchestrators.message import MessageOrchestratorApp

LOGGER_NAME = "mlp.orchestrators.pipeline"
DOMAIN = "pipeline"
PIPELINE_DIRNAME = "pipeline"

KEY_RESULTS = "results"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "out_dir": PIPELINE_DIRNAME,
    "active": [],
    "cv": {},
    "policy": {},
    "pipeline": [],
}


class PipelineOrchestrator(LoggerMixin):
    @log_call("pipeline.__init__")
    def __init__(  # noqa: PLR0913
        self,
        cfg: PipelineConfig | dict[str, Any],
        project_dir: str,
        random_state: int,
        logger_manager: SupportsGetLogger | None = None,
        out_dir: str | None = None,
        ctx: dict[str, str] | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
    ) -> None:
        self.cfg = cfg if isinstance(cfg, dict) else cfg.model_dump()
        self.random_state = random_state
        self.ctx = ctx or {}
        cfg_out = getattr(SimpleNamespace(**self.cfg), "out_dir", None)

        base_dir = Path(self.ctx["project_dir"]) if self.ctx.get("project_dir") else Path(project_dir)
        if out_dir:
            self.out_dir = Path(out_dir)
        elif cfg_out:
            p = Path(cfg_out)
            self.out_dir = p if p.is_absolute() else base_dir / p
        else:
            self.out_dir = base_dir / PIPELINE_DIRNAME
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager:
            self._init_logger(logger_manager)

        self.msg: MessageOrchestratorApp | None = message_orchestrator

    @classmethod
    @log_call("pipeline.bootstrap")
    def bootstrap(  # noqa: PLR0913
        cls,
        *,
        context_provider,
        project_dir: str,
        random_state: int,
        logger_manager: SupportsGetLogger | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
        out_dir: str | None = None,
        ini_filenames: tuple[str, ...] = ("pipeline.ini", "default.ini"),
    ) -> "PipelineOrchestrator":
        def factory(params: dict[str, Any]) -> "PipelineOrchestrator":
            ctx = params.pop("_ctx", {})
            return cls(
                cfg=params,
                project_dir=project_dir,
                random_state=random_state,
                logger_manager=logger_manager,
                out_dir=out_dir,
                ctx=ctx,
                message_orchestrator=message_orchestrator,
            )

        def validator(inst: "PipelineOrchestrator") -> None:
            if not inst.cfg.get("enabled", True):
                return
            try:
                _ = list(inst.cfg.get("pipeline", []) or [])
            except Exception:
                inst.cfg["pipeline"] = []

        def wrapped_context_provider(_name: str) -> dict[str, Any] | None:
            ctx = context_provider("pipeline") or {}
            params = (
                dict(ctx.get("orchestrators", {}).get("pipeline", {}))
                if isinstance(ctx.get("orchestrators"), dict)
                else {}
            )
            params["_ctx"] = ctx
            return params

        return bootstrap_instance(
            name="pipeline",
            factory=factory,
            defaults=DEFAULTS,
            validator=validator,
            context_provider=wrapped_context_provider,
            ini_filenames=ini_filenames,
        )

    @log_call("pipeline.attach_message")
    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        self.msg = msg

    @log_call("pipeline._filter_active_specs")
    def _filter_active_specs(self) -> list[dict[str, Any]]:
        active = set(self.cfg.get("active", []) or [])
        specs: list[dict[str, Any]] = []
        for spec in self.cfg.get("pipeline", []) or []:
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
            if not sdict.get("enabled", True):
                continue
            if active and sdict.get("name") not in active:
                continue
            specs.append(sdict)
        return specs

    @log_call("pipeline.run")
    def run(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        if not self.cfg.get("enabled", True):
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINE_DISABLED)
            return {KEY_RESULTS: []}

        specs = self._filter_active_specs()
        if self.msg:
            self.msg.emit(DOMAIN, PIPELINE_START, out_dir=str(self.out_dir), count=len(specs))

        results: list[dict[str, Any]] = []
        cv_cfg = self.cfg.get("cv", {}) or {}
        global_policy = self.cfg.get("policy", {}) or {}

        evaluator = PipelineEvaluator(
            out_dir=str(self.out_dir),
            random_state=self.random_state,
            mlflow_enabled=False,
            logger_manager=getattr(self, "logger", None),
        )

        for sdict in specs:
            name = sdict.get("name", "pipeline")
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINE_EVAL_START, name=name)
            res = evaluator.evaluate(x, y, sdict, cv_cfg, global_policy)
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINE_EVAL_DONE, name=res.get("name"), best_score=res.get("best_score"))
            results.append(res)

        if self.msg:
            self.msg.emit(DOMAIN, PIPELINE_DONE, count=len(results))
        return {KEY_RESULTS: results}
