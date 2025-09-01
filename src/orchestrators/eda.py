from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

from ..config.schemas import EDAConfig
from ..datanalysis.eda_profile import EDAProfile
from ..datanalysis.eda_summary import EDASummary
from ..instrumentation.logger_manager import LoggerManager  # interface type
from ..instrumentation.logger_mixin import LoggerMixin


class EDAOrchestrator(LoggerMixin):
    """Run EDA: YData profile + JSON synthesis, with structured logging."""

    EDA_DIR = "eda"
    KEY_PROFILE_PATH = "profile_path"
    KEY_SUMMARY_PATH = "summary_path"
    KEY_SUMMARY_DATA = "summary_data"
    KEY_FLAGS = "flags"
    LOGGER_NAME = "mlp.orchestrators.eda"

    def __init__(self, cfg: EDAConfig, project_dir: str, lm: LoggerManager) -> None:
        """Initialize EDA orchestrator.

        Args:
            cfg: EDA configuration section.
            project_dir: Project artifacts root.
            lm: Logger manager instance (stdlib or structlog backend).
        """
        self.cfg = cfg
        self.project_dir = project_dir
        self.out_dir = os.path.join(project_dir, self.EDA_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self._init_logger(lm)

    def run(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Execute EDA and return artifacts and summary."""
        n_rows, n_cols = X.shape
        self.log.info(
            "eda_start",
            extra={"extra_fields": {"out_dir": self.out_dir, "n_rows": n_rows, "n_cols": n_cols}},
        )
        df = pd.concat([X, y.rename("target")] if y is not None else [X], axis=1)
        profile_path = EDAProfile.generate_profile(
            df, self.out_dir, minimal=bool(self.cfg.profile.get("minimal", False))
        )
        summary_path, summary_data, flags = EDASummary.summarize(X, y, self.out_dir)
        self.log.info(
            "eda_done",
            extra={
                "extra_fields": {
                    "profile_path": profile_path,
                    "summary_path": summary_path,
                    "flags": flags,
                }
            },
        )
        return {
            self.KEY_PROFILE_PATH: profile_path,
            self.KEY_SUMMARY_PATH: summary_path,
            self.KEY_SUMMARY_DATA: summary_data,
            self.KEY_FLAGS: flags,
        }
