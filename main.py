from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from src.instrumentation.messages_taxonomy import APP_DONE, APP_START
from src.orchestrators.app import AppOrchestrator
from src.orchestrators.general import GeneralOrchestrator


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point: bootstraps logging and configuration via AppOrchestrator,
    then delegates to GeneralOrchestrator for application flow.
    """
    app = AppOrchestrator(cfg)
    lm = app.logger_manager
    msg = app.message_orchestrator

    log = lm.get_logger("__main__")
    msg.emit("app", APP_START, entry="main", log_file=app.logger_manager.cfg.file_path)
    try:
        go = GeneralOrchestrator(
            app.config_manager,
            logger_manager=lm,
            message_orchestrator=msg,
            ctx=app.ctx,
        )
        results = go.run()
        msg.emit("app", APP_DONE, orchestrators_run=list(results.keys()),
         report_artifacts=results.get("report", {}).get("artifacts"))
    except Exception as exc:  # noqa: BLE001
        msg.emit("app", "app_failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
