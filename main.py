from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Décorateurs: import robuste avec fallback no-op (préserve l'exécution CLI/CI)
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

from src.instrumentation.message_taxonomy import APP_DONE, APP_START
from src.orchestrators.app import AppOrchestrator
from src.orchestrators.general import GeneralOrchestrator


@hydra.main(config_path="conf", config_name="config", version_base=None)
@log_call("app.main")
def main(cfg: DictConfig) -> None:
    """
    Entry point: bootstraps logging and configuration via AppOrchestrator,
    then delegates to GeneralOrchestrator for the application flow.
    """
    # 1) Initialisation application (logging + config)
    app = AppOrchestrator(cfg)
    lm = app.logger_manager
    msg = app.message_orchestrator

    # Entrée de démarrage (en plus de logger_ready)
    lm.get_logger("__main__").info(
        "app_start",
        entry="main",
        log_file=getattr(lm, "cfg", None).file_path if getattr(lm, "cfg", None) else None,
    )

    # 2) Dump de contrôle de la config Hydra résolue (switches + params clés)
    # Utile pour vérifier que les enabled et target_column/out_dir sont bien pris.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    orch = resolved.get("orchestrators") or {}
    data_cfg = orch.get("data") or {}
    pipes_cfg = orch.get("pipeline") or {}
    msg.emit(
        "app",
        "config_resolved",
        file_enabled=(orch.get("file") or {}).get("enabled"),
        data_enabled=(orch.get("data") or {}).get("enabled"),
        eda_enabled=(orch.get("eda") or {}).get("enabled"),
        pipeline_enabled=(orch.get("pipeline") or {}).get("enabled"),
        report_enabled=(orch.get("report") or {}).get("enabled"),
        data_target_column=data_cfg.get("target_column"),
        data_auto_detect_target=data_cfg.get("auto_detect_target"),
        pipeline_out_dir=pipes_cfg.get("out_dir"),
        ctx=app.ctx,
    )

    # 3) Démarrage de l'application (log)
    msg.emit("app", APP_START, entry="main", log_file=app.logger_manager.cfg.file_path)

    # 4) Orchestration principale
    try:
        go = GeneralOrchestrator(
            app.config_manager,
            logger_manager=lm,
            message_orchestrator=msg,
            ctx=app.ctx,
        )
        results = go.run()
        msg.emit(
            "app",
            APP_DONE,
            orchestrators_run=list(results.keys()),
            report_artifacts=results.get("report", {}).get("artifacts"),
        )
    except Exception as exc:  # noqa: BLE001
        msg.emit("app", "app_failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
