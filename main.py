from __future__ import annotations

# Standard library
from pathlib import Path

# Third-party
import hydra
from omegaconf import DictConfig

# Local
from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.orchestrators.general import GeneralOrchestrator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for MLP application."""
    # 1) Load and validate configuration
    cfg_mgr = ConfigManager(cfg)
    app_cfg = cfg_mgr.load()

    # 2) Setup logging
    logger_settings = cfg_mgr.build_logger_settings()
    if logger_settings.file_path:
        Path(logger_settings.file_path).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )

    lm = build_logger_manager(logger_settings)
    lm.configure()
    log = lm.get_logger(__name__)

    log.info(
        "app_start",
        extra={"extra_fields": {"entry": "main", "log_file": logger_settings.file_path}},
    )

    try:
        # 3) Initialize and run orchestrator
        go = GeneralOrchestrator(cfg_mgr, logger_manager=lm)

        # The orchestrator will auto-detect the mode:
        # - File mode if file orchestrator is enabled
        # - Example data mode as fallback
        results = go.run()

        log.info(
            "app_done",
            extra={
                "extra_fields": {
                    "orchestrators_run": list(results.keys()),
                    "report_artifacts": results.get("report", {}).get("artifacts"),
                }
            },
        )

    except Exception as exc:
        log.error("app_failed", extra={"extra_fields": {"error": str(exc)}})
        raise


if __name__ == "__main__":
    main()
