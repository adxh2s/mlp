from __future__ import annotations

import os
import time

import pandas as pd
from ydata_profiling import ProfileReport


class EDAProfile:
    FILE_PREFIX = "profile_"
    FILE_EXT = ".html"
    DEFAULT_TITLE = "EDA Profile"

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def generate_profile(
        df: pd.DataFrame, out_dir: str, minimal: bool = False, title: str = DEFAULT_TITLE
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        prof = ProfileReport(df, title=title, minimal=minimal)
        path = os.path.join(
            out_dir, f"{EDAProfile.FILE_PREFIX}{EDAProfile._ts()}{EDAProfile.FILE_EXT}"
        )
        prof.to_file(path)
        return path
