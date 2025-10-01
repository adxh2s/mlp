from __future__ import annotations

"""EDAProfile: HTML profiling export with structured telemetry."""

# Décorateurs: import robuste avec fallback no-op
try:
    from decorators import log_call
except Exception:  # pragma: no cover
    from typing import Callable, TypeVar, ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

    def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:  # type: ignore[override]
        def deco(fn: Callable[P, T]) -> Callable[P, T]:
            return fn
        return deco

import os
import time

import pandas as pd
from ydata_profiling import ProfileReport


class EDAProfile:
    """Thin wrapper over ydata_profiling for consistent artifacts naming."""

    FILE_PREFIX = "profile_"
    FILE_EXT = ".html"
    DEFAULT_TITLE = "EDA Profile"

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    @log_call("eda_profile.generate_profile")
    def generate_profile(
        df: pd.DataFrame,
        out_dir: str,
        minimal: bool = False,
        title: str = DEFAULT_TITLE,
    ) -> str:
        """Generate an HTML profile report and return its path."""
        os.makedirs(out_dir, exist_ok=True)
        prof = ProfileReport(df, title=title, minimal=minimal)
        path = os.path.join(out_dir, f"{EDAProfile.FILE_PREFIX}{EDAProfile._ts()}{EDAProfile.FILE_EXT}")
        prof.to_file(path)
        return path
