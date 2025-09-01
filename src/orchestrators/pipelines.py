from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config.schemas import PipelinesConfig
from ..instrumentation.logger_manager import LoggerManager  # interface type
from ..instrumentation.logger_mixin import LoggerMixin
from ..modeling.evaluator import PipelineEvaluator


class PipelineOrchestrator(LoggerMixin):
    """Build and evaluate declared pipelines with structured logging."""

    PIPELINES_DIR = "pipelines"
    KEY_RESULTS = "results"
    LOGGER_NAME = "mlp.orchestrators.pipelines"

    def __init__(
        self,
        cfg: PipelinesConfig,
        project_dir: str,
        random_state: int,
        logger_manager: Optional[LoggerManager] = None,
        out_dir: Optional[str] = None,
    ) -> None:
        """Initialize pipelines orchestrator.

        Args:
            cfg: Pipelines configuration section.
            project_dir: Project artifacts root.
            random_state: Random seed for CV/reproducibility.
            logger_manager: Logger manager instance (optional).
            out_dir: Optional explicit output directory for pipeline artifacts.
        """
        self.cfg = cfg
        # Par défaut, conserver l'ancien chemin: <project_dir>/pipelines
        self.out_dir = out_dir if out_dir else os.path.join(project_dir, self.PIPELINES_DIR)
        self.random_state = random_state
        os.makedirs(self.out_dir, exist_ok=True)

        # Init logger (manager si fourni, sinon fallback stdlib)
        self.lm = logger_manager
        if self.lm is not None:
            self._init_logger(self.lm)
        else:
            self.log = logging.getLogger(self.LOGGER_NAME)

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate configured pipelines and return summarized results."""
        if not self.cfg.enabled:
            self.log.info("pipelines_disabled")
            return {self.KEY_RESULTS: []}

        n_rows, n_cols = X.shape
        self.log.info(
            "pipelines_start",
            extra={"extra_fields": {"out_dir": self.out_dir, "n_rows": n_rows, "n_cols": n_cols}},
        )

        evaluator = PipelineEvaluator(
            out_dir=self.out_dir,
            random_state=self.random_state,  # <-- correction ici
            mlflow_enabled=False,
            logger_manager=self.lm,
        )

        results: List[Dict[str, Any]] = []
        for spec in self.cfg.pipelines:
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else spec
            self.log.info(
                "pipeline_eval_start", extra={"extra_fields": {"name": sdict.get("name")}}
            )
            res = evaluator.evaluate(X, y, sdict, self.cfg.cv)
            self.log.info(
                "pipeline_eval_done",
                extra={
                    "extra_fields": {"name": res.get("name"), "best_score": res.get("best_score")}
                },
            )
            results.append(res)

        self.log.info("pipelines_done", extra={"extra_fields": {"count": len(results)}})
        return {self.KEY_RESULTS: results}
