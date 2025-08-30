from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from ..config.schemas import PipelinesConfig
from ..instrumentation.logger_mixin import LoggerMixin
from ..instrumentation.logger_manager import LoggerManager  # interface type
from ..modeling.evaluator import PipelineEvaluator


class PipelineOrchestrator(LoggerMixin):
    """Build and evaluate declared pipelines with structured logging."""

    PIPELINES_DIR = "pipelines"
    KEY_RESULTS = "results"
    LOGGER_NAME = "mlp.orchestrators.pipelines"

    def __init__(self, cfg: PipelinesConfig, project_dir: str, random_state: int, lm: LoggerManager) -> None:
        """Initialize pipelines orchestrator.

        Args:
            cfg: Pipelines configuration section.
            project_dir: Project artifacts root.
            random_state: Random seed for CV/reproducibility.
            lm: Logger manager instance.
        """
        self.cfg = cfg
        self.out_dir = os.path.join(project_dir, self.PIPELINES_DIR)
        self.random_state = random_state
        os.makedirs(self.out_dir, exist_ok=True)
        self._init_logger(lm)

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate configured pipelines and return summarized results."""
        if not self.cfg.enabled:
            self.log.info("pipelines_disabled")
            return {self.KEY_RESULTS: []}

        self.log.info(
            "pipelines_start",
            extra={"extra_fields": {"out_dir": self.out_dir, "n_rows": int(X.shape), "n_cols": int(X.shape)}},
        )
        evaluator = PipelineEvaluator(self.out_dir, self.random_state)
        results: List[Dict[str, Any]] = []
        for spec in self.cfg.pipelines:
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else spec
            self.log.info("pipeline_eval_start", extra={"extra_fields": {"name": sdict.get("name")}})
            res = evaluator.evaluate(X, y, sdict, self.cfg.cv)
            self.log.info(
                "pipeline_eval_done",
                extra={"extra_fields": {"name": res.get("name"), "best_score": res.get("best_score")}},
            )
            results.append(res)

        self.log.info("pipelines_done", extra={"extra_fields": {"count": len(results)}})
        return {self.KEY_RESULTS: results}
