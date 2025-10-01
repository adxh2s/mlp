from __future__ import annotations

import os
from typing import Any, cast

# Décorateurs: import robuste avec fallback no-op
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

import pandas as pd

from src.config.schemas import EDAConfig
from src.datanalysis.eda_profile import EDAProfile
from src.datanalysis.eda_summary import EDASummary
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.message_taxonomy import EDA_DONE, EDA_START
from src.orchestrators.bootstrap import bootstrap_instance
from src.orchestrators.message import MessageOrchestratorApp

EDA_DIR = "eda"

KEY_PROFILE_PATH = "profile_path"
KEY_SUMMARY_PATH = "summary_path"
KEY_SUMMARY_DATA = "summary_data"
KEY_FLAGS = "flags"

LOGGER_NAME = "mlp.orchestrators.eda"
DOMAIN = "eda"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "profile": {"minimal": False, "title": "EDA Profile"},
    "out_dir": EDA_DIR,
}


def _as_target(y: pd.Series | None) -> pd.Series | None:
    if y is None:
        return None
    if getattr(y, "name", None) == "target":
        return y
    y2 = y.copy(deep=False)
    try:
        y2.name = "target"
    except Exception:
        y2 = y.copy(deep=True)
        y2.name = "target"
    return y2


class EDAOrchestrator(LoggerMixin):
    """Run EDA: profile and summary, and emit localized events."""

    @log_call("eda.__init__")
    def __init__(
        self,
        cfg: EDAConfig | dict[str, Any],
        project_dir: str,
        logger_manager: LoggerManager | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
    ) -> None:
        self.cfg = cfg if isinstance(cfg, dict) else cfg.model_dump()
        self.project_dir = project_dir
        self.out_dir = os.path.join(project_dir, str(self.cfg.get("out_dir", EDA_DIR)))
        os.makedirs(self.out_dir, exist_ok=True)

        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager:
            self._init_logger(cast(Any, logger_manager))
        self.msg = message_orchestrator

    @classmethod
    @log_call("eda.bootstrap")
    def bootstrap(
        cls,
        *,
        context_provider,
        project_dir: str,
        logger_manager: LoggerManager | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
        ini_filenames: tuple[str, ...] = ("eda.ini", "default.ini"),
    ) -> "EDAOrchestrator":
        def factory(params: dict[str, Any]) -> "EDAOrchestrator":
            return cls(
                params,
                project_dir=project_dir,
                logger_manager=logger_manager,
                message_orchestrator=message_orchestrator,
            )

        def validator(inst: "EDAOrchestrator") -> None:
            if not inst.cfg.get("enabled", True):
                return
            os.makedirs(inst.out_dir, exist_ok=True)

        return bootstrap_instance(
            name="eda",
            factory=factory,
            defaults=DEFAULTS,
            validator=validator,
            context_provider=context_provider,
            ini_filenames=ini_filenames,
        )

    @log_call("eda.attach_message")
    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        self.msg = msg

    @log_call("eda.run")
    def run(self, x: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        if not self.cfg.get("enabled", True):
            return {}

        n_rows, n_cols = x.shape
        if self.msg:
            self.msg.emit(DOMAIN, EDA_START, out_dir=self.out_dir, n_rows=n_rows, n_cols=n_cols)

        parts: list[pd.DataFrame | pd.Series] = [x]
        y_named = _as_target(y)
        if y_named is not None:
            parts.append(y_named)
        df = pd.concat(parts, axis=1)

        prof_min = bool(self.cfg.get("profile", {}).get("minimal", False))
        prof_title = str(self.cfg.get("profile", {}).get("title", "EDA Profile"))
        profile_path = EDAProfile.generate_profile(df, self.out_dir, minimal=prof_min, title=prof_title)

        summary_path, summary_data, flags = EDASummary.summarize(x, y, self.out_dir)

        if self.msg:
            self.msg.emit(DOMAIN, EDA_DONE, profile_path=str(profile_path), summary_path=str(summary_path), flags=flags)

        return {
            KEY_PROFILE_PATH: str(profile_path),
            KEY_SUMMARY_PATH: str(summary_path),
            KEY_SUMMARY_DATA: summary_data,
            KEY_FLAGS: flags,
        }
