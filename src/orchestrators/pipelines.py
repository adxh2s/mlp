from __future__ import annotations

"""
Orchestrateur de pipelines: construit et évalue les pipelines déclarés,
en émettant des événements structurés via un MessageOrchestrator partagé.
"""

from pathlib import Path
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
from src.modeling.pipelines.evaluator import PipelineEvaluator
from src.orchestrators.messages import MessageOrchestrator

# Constantes
LOGGER_NAME = "mlp.orchestrators.pipelines"
DOMAIN = "pipelines"
PIPELINES_DIRNAME = "pipelines"
KEY_RESULTS = "results"


class PipelineOrchestrator(LoggerMixin):
    """
    Orchestrateur des pipelines ML.

    - Filtre les pipelines actifs (enabled + active list).
    - Délègue la construction/évaluation à PipelineEvaluator.
    - Produit des événements localisés (MessageOrchestrator).
    """

    def __init__(
        self,
        cfg: PipelinesConfig,
        project_dir: str,
        random_state: int,
        logger_manager: Optional[LoggerManager] = None,
        out_dir: Optional[str] = None,
        cfg_mgr: Optional[Any] = None,
        ctx: Optional[dict[str, str]] = None,
    ) -> None:
        self.cfg = cfg
        self.random_state = random_state
        self.ctx = ctx or {}

        # Sortie des artefacts (priorité: arg -> YAML -> défaut)
        cfg_out = getattr(self.cfg, "out_dir", None)
        base_dir = Path(self.ctx["project_dir"]) if self.ctx.get("project_dir") else Path(project_dir)

        if out_dir:
            self.out_dir = Path(out_dir)
        elif cfg_out:
            p = Path(cfg_out)
            self.out_dir = p if p.is_absolute() else base_dir / p
        else:
            self.out_dir = base_dir / PIPELINES_DIR  # constante déjà définie dans ce module

        self.out_dir.mkdir(parents=True, exist_ok=True)


        # Logging
        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager:
            self._init_logger(logger_manager)

        # Messages (injecté par GeneralOrchestrator)
        self.msg: Optional[MessageOrchestrator] = None

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attache l’émetteur de messages localisés à l’orchestrateur."""
        self.msg = msg

    def _filter_active_specs(self) -> List[Dict[str, Any]]:
        active = set(getattr(self.cfg, "active", []) or [])
        specs = []
        for spec in self.cfg.pipelines:
            # Pydantic -> dict
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
            if not sdict.get("enabled", True):
                continue
            if active and sdict.get("name") not in active:
                continue
            specs.append(sdict)
        return specs

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Exécute les pipelines actifs et retourne la liste des résultats."""
        if not self.cfg.enabled:
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINES_DISABLED)
            return {KEY_RESULTS: []}

        specs = self._filter_active_specs()
        if self.msg:
            self.msg.emit(DOMAIN, PIPELINES_START, out_dir=str(self.out_dir), count=len(specs))

        results: List[Dict[str, Any]] = []
        cv_cfg = getattr(self.cfg, "cv", {}) or {}
        global_policy = getattr(self.cfg, "policy", {}) or {}

        evaluator = PipelineEvaluator(
            out_dir=str(self.out_dir),
            random_state=self.random_state,
            mlflow_enabled=False,
            logger_manager=getattr(self, "lm", None),
        )

        for sdict in specs:
            name = sdict.get("name", "pipeline")

            if self.msg:
                self.msg.emit(DOMAIN, PIPELINES_EVAL_START, name=name)

            res = evaluator.evaluate(X, y, sdict, cv_cfg, global_policy)

            if self.msg:
                self.msg.emit(DOMAIN, PIPELINES_EVAL_DONE, name=res.get("name"), best_score=res.get("best_score"))

            results.append(res)

        if self.msg:
            self.msg.emit(DOMAIN, PIPELINES_DONE, count=len(results))

        return {KEY_RESULTS: results}
