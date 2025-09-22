from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.schemas import PipelinesConfig
from src.instrumentation.logger_mixin import LoggerMixin, SupportsGetLogger
from src.instrumentation.messages_taxonomy import (
    PIPELINES_DISABLED,
    PIPELINES_DONE,
    PIPELINES_EVAL_DONE,
    PIPELINES_EVAL_START,
    PIPELINES_START,
)
from src.modeling.pipelines.evaluator import PipelineEvaluator
from src.orchestrators.messages import MessageOrchestrator

"""
Orchestrateur des pipelines ML/DL:
- Filtre les pipelines actifs (enabled + liste active).
- Délègue l'évaluation à PipelineEvaluator (ML/DL).
- Émet des événements structurés via MessageOrchestrator.
"""


# Constantes module
LOGGER_NAME = "mlp.orchestrators.pipelines"
DOMAIN = "pipelines"
PIPELINES_DIRNAME = "pipelines"
KEY_RESULTS = "results"


class PipelineOrchestrator(LoggerMixin):
    """
    Exécute les pipelines déclarés et agrège leurs résultats, avec journalisation et événements. 
    - La sortie des artefacts est résolue par priorité: out_dir argument > YAML > défaut projet/“pipelines”. 
    - Le logger est initialisé via un provider conforme à SupportsGetLogger. 
    """

    def __init__(  # noqa: PLR0913
        self,
        cfg: PipelinesConfig,
        project_dir: str,
        random_state: int,
        logger_provider: SupportsGetLogger | None = None,
        out_dir: str | None = None,
        ctx: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.random_state = random_state
        self.ctx = ctx or {}

        # Résolution du répertoire de sortie
        cfg_out = getattr(self.cfg, "out_dir", None)
        base_dir = Path(self.ctx["project_dir"]) if self.ctx.get("project_dir") else Path(project_dir)
        if out_dir:
            self.out_dir = Path(out_dir)
        elif cfg_out:
            p = Path(cfg_out)
            self.out_dir = p if p.is_absolute() else base_dir / p
        else:
            self.out_dir = base_dir / PIPELINES_DIRNAME
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.LOGGER_NAME = LOGGER_NAME
        if logger_provider:
            self._init_logger(logger_provider)

        # Messages (injecté par l’orchestrateur général)
        self.msg: MessageOrchestrator | None = None

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attache l’émetteur de messages localisés (structlog) pour ce domaine."""
        self.msg = msg

    def _filter_active_specs(self) -> list[dict[str, Any]]:
        """
        Retourne les spécifications de pipelines à exécuter:
        - enabled = True
        - inclus dans la liste 'active' si elle est fournie.
        """
        active = set(getattr(self.cfg, "active", []) or [])
        specs: list[dict[str, Any]] = []
        for spec in self.cfg.pipelines:
            sdict = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
            if not sdict.get("enabled", True):
                continue
            if active and sdict.get("name") not in active:
                continue
            specs.append(sdict)
        return specs

    def run(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Construit et évalue chaque pipeline actif, en émettant des événements et en renvoyant la synthèse.
        - x: jeux de caractéristiques
        - y: cibles
        """
        if not self.cfg.enabled:
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINES_DISABLED)
            return {KEY_RESULTS: []}

        specs = self._filter_active_specs()
        if self.msg:
            self.msg.emit(DOMAIN, PIPELINES_START, out_dir=str(self.out_dir), count=len(specs))

        results: list[dict[str, Any]] = []
        cv_cfg = getattr(self.cfg, "cv", {}) or {}
        global_policy = getattr(self.cfg, "policy", {}) or {}

        evaluator = PipelineEvaluator(
            out_dir=str(self.out_dir),
            random_state=self.random_state,
            mlflow_enabled=False,
            logger_manager=None,
        )

        for sdict in specs:
            name = sdict.get("name", "pipeline")
            if self.msg:
                self.msg.emit(DOMAIN, PIPELINES_EVAL_START, name=name)

            res = evaluator.evaluate(x, y, sdict, cv_cfg, global_policy)

            if self.msg:
                self.msg.emit(
                    DOMAIN,
                    PIPELINES_EVAL_DONE,
                    name=res.get("name"),
                    best_score=res.get("best_score"),
                )
            results.append(res)

        if self.msg:
            self.msg.emit(DOMAIN, PIPELINES_DONE, count=len(results))

        return {KEY_RESULTS: results}
