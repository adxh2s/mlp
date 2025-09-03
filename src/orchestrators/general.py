from __future__ import annotations

# Standard library
import logging
import os
from typing import Any

# Third-party
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Local
from ..config.schemas import AppConfig
from ..instrumentation.config_manager import ConfigManager
from ..instrumentation.logger_factory import build_logger_manager
from ..instrumentation.logger_mixin import LoggerMixin
from .data import DataOrchestrator
from .eda import EDAOrchestrator
from .file import FileOrchestrator
from .pipelines import PipelineOrchestrator
from .report import ReportOrchestrator


class GeneralOrchestrator(LoggerMixin):
    """Coordinate file intake, data processing, EDA, pipelines, and reporting.

    Orchestrates the end-to-end flow:
    - Optional file intake via FileOrchestrator
    - Data processing via DataOrchestrator
    - EDA orchestration
    - Pipelines orchestration
    - Report orchestration
    """

    # Class constants
    KEY_FILE = "file"
    KEY_DATA = "data"
    KEY_EDA = "eda"
    KEY_PIPELINES = "pipelines"
    KEY_REPORT = "report"
    LOGGER_NAME = "mlp.orchestrators.general"

    def __init__(self, cfg_mgr: ConfigManager, logger_manager=None) -> None:
        """Initialize with validated configuration manager."""
        self.cfg_mgr = cfg_mgr
        self.cfg: AppConfig = cfg_mgr.model

        # Output directories
        self.out_dir = self.cfg.project.output_dir
        self.project_dir = os.path.join(self.out_dir, self.cfg.project.name)
        os.makedirs(self.project_dir, exist_ok=True)

        # Logger manager
        if logger_manager is not None:
            self.lm = logger_manager
        else:
            self.lm = build_logger_manager(self.cfg.logger)
            self.lm.configure()

        # Initialize logger
        try:
            self._init_logger(self.lm)
        except Exception:
            self.log = logging.getLogger(self.LOGGER_NAME)

        self.log.info("general_init", extra={"extra_fields": {"project_dir": self.project_dir}})

    def load_example_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load example dataset for development/testing."""
        ds = load_breast_cancer(as_frame=True)
        X = ds.frame.drop(columns=["target"])
        y = ds.frame["target"]
        return X, y

    def run_from_files(self) -> dict[str, Any]:
        """Pipeline mode: File → Data → EDA → Pipelines → Report."""
        results = {}
        orchestrators = self.cfg.orchestrators

        self.log.info("general_start_from_files")

        # 1) File orchestrator - Load raw data
        if orchestrators.file and orchestrators.file.enabled:
            try:
                file_orch = FileOrchestrator(orchestrators.file, logger_manager=self.lm)
                file_result = file_orch.process_input()
                results[self.KEY_FILE] = file_result

                if not file_result.get("found", False):
                    self.log.warning("no_input_files_found")
                    return results

                raw_data = file_result["data"]
            except Exception as exc:
                self.log.error(
                    "file_orchestrator_failed", extra={"extra_fields": {"error": str(exc)}}
                )
                return results
        else:
            self.log.error("file_orchestrator_disabled_but_required")
            return {}

        # 2) Data orchestrator - Process raw data
        if orchestrators.data.enabled:
            try:
                data_orch = DataOrchestrator(orchestrators.data, logger_manager=self.lm)
                data_result = data_orch.run(raw_data)
                results[self.KEY_DATA] = data_result
                X, y = data_result["X"], data_result["y"]
            except Exception as exc:
                self.log.error(
                    "data_orchestrator_failed", extra={"extra_fields": {"error": str(exc)}}
                )
                return results
        else:
            # Fallback: assume raw_data is already a DataFrame
            if isinstance(raw_data, pd.DataFrame):
                X, y = raw_data, None
            else:
                self.log.error("data_orchestrator_disabled_but_raw_data_not_dataframe")
                return results

        # 3) Continue with existing orchestrators
        return self._run_ml_orchestrators(X, y, results)

    def run_from_data(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        """Direct mode: Start with prepared data."""
        results = {
            self.KEY_DATA: {
                "X": X,
                "y": y,
                "metadata": {
                    "features_count": X.shape[1],
                    "samples_count": X.shape[0],
                    "has_target": y is not None,
                },
            }
        }
        self.log.info("general_start_from_data", extra={"extra_fields": {"shape": X.shape}})
        return self._run_ml_orchestrators(X, y, results)

    def _run_ml_orchestrators(
        self, X: pd.DataFrame, y: pd.Series | None, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute EDA/Pipelines/Report orchestrators."""
        orchestrators = self.cfg.orchestrators

        # EDA
        if orchestrators.eda.enabled:
            try:
                eda = EDAOrchestrator(orchestrators.eda, self.project_dir, self.lm)
                results[self.KEY_EDA] = eda.run(X, y)
            except Exception as exc:
                self.log.error(
                    "eda_orchestrator_failed", extra={"extra_fields": {"error": str(exc)}}
                )

        # Pipelines (only if target available)
        if orchestrators.pipelines.enabled and y is not None:
            try:
                pipes = PipelineOrchestrator(
                    orchestrators.pipelines,
                    project_dir=self.project_dir,
                    random_state=self.cfg.project.random_state,
                    logger_manager=self.lm,
                    out_dir=self.out_dir,
                )
                results[self.KEY_PIPELINES] = pipes.run(X, y)
            except Exception as exc:
                self.log.error(
                    "pipelines_orchestrator_failed", extra={"extra_fields": {"error": str(exc)}}
                )

        # Report
        if orchestrators.report.enabled:
            try:
                rep = ReportOrchestrator(orchestrators.report, self.project_dir, self.cfg, self.lm)
                results[self.KEY_REPORT] = rep.run(
                    results.get(self.KEY_EDA, {}), results.get(self.KEY_PIPELINES, {"results": []})
                )
            except Exception as exc:
                self.log.error(
                    "report_orchestrator_failed", extra={"extra_fields": {"error": str(exc)}}
                )

        self.log.info(
            "general_done",
            extra={
                "extra_fields": {
                    "report_artifacts": results.get(self.KEY_REPORT, {}).get("artifacts")
                }
            },
        )
        return results

    # Méthode de compatibilité pour l'ancien API
    def run(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> dict[str, Any]:
        """Run orchestrator - auto-detect mode based on configuration and parameters."""
        # Mode fichier si FileOrchestrator activé et pas de données fournies
        if (
            self.cfg.orchestrators.file
            and self.cfg.orchestrators.file.enabled
            and X is None
            and y is None
        ):
            return self.run_from_files()

        # Mode données directes
        if X is not None:
            return self.run_from_data(X, y)

        # Mode exemple (fallback)
        self.log.info("using_example_data")
        X_example, y_example = self.load_example_data()
        return self.run_from_data(X_example, y_example)
