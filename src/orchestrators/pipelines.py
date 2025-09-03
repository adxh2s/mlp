from __future__ import annotations

"""
Pipelines orchestrator: build and evaluate declared pipelines with
localized, structured logging using a shared MessageOrchestrator.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.schemas import PipelinesConfig
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import (
    PIPELINES_DISABLED,
    PIPELINES_DONE,
    PIPELINES_EVAL_DONE,
    PIPELINES_EVAL_START,
    PIPELINES_START,
)
from src.modeling.evaluator import PipelineEvaluator
from src.orchestrators.messages import MessageOrchestrator

# Constants
PIPELINES_DIR = "pipelines"
KEY_RESULTS = "results"
LOGGER_NAME = "mlp.orchestrators.pipelines"
DOMAIN = "pipelines"


class PipelineOrchestrator(LoggerMixin):
    """Build and evaluate configured pipelines, emitting localized events."""

    def __init__(
        self,
        cfg: PipelinesConfig,
        project_dir: str,
        random_state: int,
        logger_manager: Optional[LoggerManager] = None,
        out_dir: Optional[str] = None,
        cfg_mgr: Optional[Any] = None,
    ) -> None:
        """
        Initialize pipelines orchestrator.

        Args:
            cfg: Pipelines configuration section.
            project_dir: Project artifacts root.
            random_state: Random seed for CV/reproducibility.
            logger_manager: Logger manager instance (optional).
            out_dir: Optional explicit output directory for artifacts.
            cfg_mgr: Optional ConfigManager for message orchestrator creation.
        """
        self.cfg = cfg
        self.out_dir = out_dir if out_dir else os.path.join(project_dir, PIPELINES_DIR)
        self.random_state = random_state
        os.makedirs(self.out_dir, exist_ok=True)

        self.lm = logger_manager
        self.LOGGER_NAME = LOGGER_NAME
        if self.lm is not None:
            self._init_logger(self.lm)
        else:
            self.log = logging.getLogger(LOGGER_NAME)

        # Message orchestrator (injected by GeneralOrchestrator via attach_messages)
        self.msg: Optional[MessageOrchestrator] = None

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attach a MessageOrchestrator for localized emissions."""
        self.msg = msg

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate configured pipelines and return summarized results."""
        if not self.cfg.enabled:
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINES_DISABLED)
            else:
                self.log.info("pipelines_disabled")
            return {KEY_RESULTS: []}

        n_rows, n_cols = X.shape
        if self.msg:
            self.msg.emit(DOMAIN, PIPELINES_START, out_dir=self.out_dir, n_rows=n_rows, n_cols=n_cols)
        else:
            self.log.info(
                "pipelines_start",
                extra={
                    "extra_fields": {
                        "out_dir": self.out_dir,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                    }
                },
            )

        evaluator = PipelineEvaluator(
            out_dir=self.out_dir,
            random_state=self.random_state,
            mlflow_enabled=False,
            logger_manager=self.lm,
        )

        results: List[Dict[str, Any]] = []
        for spec in self.cfg.pipelines:
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else spec
            name = sdict.get("name")
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINE_EVAL_START, name=name)
            else:
                self.log.info("pipeline_eval_start", extra={"extra_fields": {"name": name}})

            res = evaluator.evaluate(X, y, sdict, self.cfg.cv)

            best_score = res.get("best_score")
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINE_EVAL_DONE, name=res.get("name"), best_score=best_score)
            else:
                self.log.info(
                    "pipeline_eval_done",
                    extra={"extra_fields": {"name": res.get("name"), "best_score": best_score}},
                )

            results.append(res)

        if self.msg:
            self.msg.emit(DOMAIN, PIPELINES_DONE, count=len(results))
        else:
            self.log.info("pipelines_done", extra={"extra_fields": {"count": len(results)}})

        return {KEY_RESULTS: results}
