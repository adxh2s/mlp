from __future__ import annotations

from pathlib import Path
from typing import Any

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
from src.orchestrators.message import MessageOrchestratorApp  # alignement app-level

LOGGER_NAME = "mlp.orchestrators.pipeline"
DOMAIN = "pipeline"
PIPELINE_DIRNAME = "pipeline"
KEY_RESULTS = "results"


class PipelineOrchestrator(LoggerMixin):
    def __init__(  # noqa: PLR0913
        self,
        cfg: PipelineConfig,
        project_dir: str,
        random_state: int,
        logger_manager: SupportsGetLogger | None = None,
        out_dir: str | None = None,
        ctx: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.random_state = random_state
        self.ctx = ctx or {}

        cfg_out = getattr(self.cfg, "out_dir", None)
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

        self.msg: MessageOrchestratorApp | None = None

    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        self.msg = msg

    def _filter_active_specs(self) -> list[dict[str, Any]]:
        active = set(getattr(self.cfg, "active", []) or [])
        specs: list[dict[str, Any]] = []
        for spec in self.cfg.pipeline:
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
            if not sdict.get("enabled", True):
                continue
            if active and sdict.get("name") not in active:
                continue
            specs.append(sdict)
        return specs

    def run(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        if not self.cfg.enabled:
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINE_DISABLED)
            return {KEY_RESULTS: []}

        specs = self._filter_active_specs()
        if self.msg:
            self.msg.emit(DOMAIN, PIPELINE_START, out_dir=str(self.out_dir), count=len(specs))

        results: list[dict[str, Any]] = []

        cv_cfg = getattr(self.cfg, "cv", {}) or {}
        global_policy = getattr(self.cfg, "policy", {}) or {}

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
