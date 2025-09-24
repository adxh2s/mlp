from __future__ import annotations

import os
from typing import Any, cast

import pandas as pd

from src.config.schemas import EDAConfig
from src.datanalysis.eda_profile import EDAProfile
from src.datanalysis.eda_summary import EDASummary
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import EDA_DONE, EDA_START
from src.orchestrators.messages import MessagesOrchestrator

"""
EDA orchestrator: YData profile and JSON summary with structured logs.
"""

# Constants
EDA_DIR = "eda"
KEY_PROFILE_PATH = "profile_path"
KEY_SUMMARY_PATH = "summary_path"
KEY_SUMMARY_DATA = "summary_data"
KEY_FLAGS = "flags"
LOGGER_NAME = "mlp.orchestrators.eda"
DOMAIN = "eda"


def _as_target(y: pd.Series | None) -> pd.Series | None:
    """
    Retourne y avec name='target' sans utiliser Series.rename pour éviter les surcharges typées.
    """
    if y is None:
        return None
    if getattr(y, "name", None) == "target":
        return y
    y2 = y.copy(deep=False)
    try:
        y2.name = "target"
    except Exception:
        # Si l’attribut est verrouillé, retomber sur une copie matérielle
        y2 = y.copy(deep=True)
        y2.name = "target"
    return y2


class EDAOrchestrator(LoggerMixin):
    """Run EDA: profile and summary, and emit localized events."""

    def __init__(self, cfg: EDAConfig, project_dir: str, logger_manager: LoggerManager) -> None:
        self.cfg = cfg
        self.project_dir = project_dir
        self.out_dir = os.path.join(project_dir, EDA_DIR)
        os.makedirs(self.out_dir, exist_ok=True)

        self.LOGGER_NAME = LOGGER_NAME
        self._init_logger(cast(Any, logger_manager))
        self.log: Any = getattr(self, "log", None)

        self.msg: MessagesOrchestrator | None = None

    def attach_messages(self, msg: MessagesOrchestrator) -> None:
        """Attach a MessagesOrchestrator for localized emissions."""
        self.msg = msg

    def run(self, x: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        """Execute EDA and return artifacts and summary."""
        n_rows, n_cols = x.shape
        if self.msg:
            self.msg.emit(DOMAIN, EDA_START, out_dir=self.out_dir, n_rows=n_rows, n_cols=n_cols)
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("eda_start", extra={"extra_fields": {"out_dir": self.out_dir, "n_rows": n_rows, "n_cols": n_cols}})

        parts: list[pd.DataFrame | pd.Series] = [x]
        y_named = _as_target(y)
        if y_named is not None:
            parts.append(y_named)
        df = pd.concat(parts, axis=1)

        profile_path = EDAProfile.generate_profile(df, self.out_dir, minimal=bool(self.cfg.profile.get("minimal", False)))
        summary_path, summary_data, flags = EDASummary.summarize(x, y, self.out_dir)

        if self.msg:
            self.msg.emit(DOMAIN, EDA_DONE, profile_path=str(profile_path), summary_path=str(summary_path), flags=flags)
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info(
                    "eda_done",
                    extra={"extra_fields": {"profile_path": str(profile_path), "summary_path": str(summary_path), "flags": flags}},
                )

        return {
            KEY_PROFILE_PATH: str(profile_path),
            KEY_SUMMARY_PATH: str(summary_path),
            KEY_SUMMARY_DATA: summary_data,
            KEY_FLAGS: flags,
        }
