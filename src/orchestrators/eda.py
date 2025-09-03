from __future__ import annotations

"""
EDA orchestrator: YData profile and JSON summary with structured logs.
"""

import os
from typing import Any, Dict, Optional

import pandas as pd

from src.config.schemas import EDAConfig
from src.datanalysis.eda_profile import EDAProfile
from src.datanalysis.eda_summary import EDASummary
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import EDA_DONE, EDA_START
from src.orchestrators.messages import MessageOrchestrator

# Constants
EDA_DIR = "eda"
KEY_PROFILE_PATH = "profile_path"
KEY_SUMMARY_PATH = "summary_path"
KEY_SUMMARY_DATA = "summary_data"
KEY_FLAGS = "flags"
LOGGER_NAME = "mlp.orchestrators.eda"
DOMAIN = "eda"


class EDAOrchestrator(LoggerMixin):
    """Run EDA: profile and summary, and emit localized events."""

    def __init__(self, cfg: EDAConfig, project_dir: str, lm: LoggerManager) -> None:
        """
        Initialize EDA orchestrator.

        Args:
            cfg: EDA configuration section.
            project_dir: Project artifacts root.
            lm: Logger manager instance (stdlib or structlog).
        """
        self.cfg = cfg
        self.project_dir = project_dir
        self.out_dir = os.path.join(project_dir, EDA_DIR)
        os.makedirs(self.out_dir, exist_ok=True)

        self.LOGGER_NAME = LOGGER_NAME
        self._init_logger(lm)

        # Message orchestrator derived from the same logger/config
        # It is the caller's responsibility to pass a properly initialized
        # MessageOrchestrator if needed. Here, for simplicity, we build one
        # from lm's config path via an optional pattern. In practice, prefer
        # passing a shared instance from GeneralOrchestrator.
        self.msg: Optional[MessageOrchestrator] = None  # may be injected

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attach a MessageOrchestrator for localized emissions."""
        self.msg = msg

    def run(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Execute EDA and return artifacts and summary."""
        n_rows, n_cols = X.shape

        if self.msg:
            self.msg.emit(DOMAIN, EDA_START, out_dir=self.out_dir, n_rows=n_rows, n_cols=n_cols)
        else:
            self.log.info(
                "eda_start",
                extra={
                    "extra_fields": {"out_dir": self.out_dir, "n_rows": n_rows, "n_cols": n_cols}
                },
            )

        df = pd.concat([X, y.rename("target")] if y is not None else [X], axis=1)
        profile_path = EDAProfile.generate_profile(
            df, self.out_dir, minimal=bool(self.cfg.profile.get("minimal", False))
        )

        summary_path, summary_data, flags = EDASummary.summarize(X, y, self.out_dir)

        if self.msg:
            self.msg.emit(
                DOMAIN,
                EDA_DONE,
                profile_path=str(profile_path),
                summary_path=str(summary_path),
                flags=flags,
            )
        else:
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
            KEY_PROFILE_PATH: profile_path,
            KEY_SUMMARY_PATH: summary_path,
            KEY_SUMMARY_DATA: summary_data,
            KEY_FLAGS: flags,
        }
