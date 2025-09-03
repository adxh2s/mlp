from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd


class EDASummary:
    FILE_PREFIX = "eda_summary_"
    FILE_EXT = ".json"
    HIGH_CORR_THRESHOLD = 0.95
    IMBALANCE_THRESHOLD = 0.8
    HIGH_DIM_N_RATIO = 5

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def summarize(
        X: pd.DataFrame, y: pd.Series | None, out_dir: str
    ) -> tuple[str, dict[str, Any], dict[str, bool]]:
        os.makedirs(out_dir, exist_ok=True)
        n, p = X.shape
        na_counts = X.isna().sum().sort_values(ascending=False)
        dup_count = int(X.duplicated().sum())
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        corr = X[numeric_cols].corr().abs() if numeric_cols else pd.DataFrame()
        max_offdiag = (
            float(corr.where(~np.eye(len(corr), dtype=bool)).max().max()) if not corr.empty else 0.0
        )
        high_collinearity = max_offdiag >= EDASummary.HIGH_CORR_THRESHOLD

        class_imbalance = False
        y_stats = None
        if y is not None:
            y_dist = y.value_counts(normalize=True)
            y_stats = y_dist.to_dict()
            class_imbalance = y_dist.max() >= EDASummary.IMBALANCE_THRESHOLD

        needs_scaling = True
        high_dimensional = n < EDASummary.HIGH_DIM_N_RATIO * p

        summary = {
            "shape": {"n_samples": n, "n_features": p},
            "na_top10": na_counts.head(10).to_dict(),
            "duplicates": dup_count,
            "numeric_features": len(numeric_cols),
            "max_abs_corr_offdiag": max_offdiag,
            "y_distribution": y_stats,
        }
        path = os.path.join(
            out_dir, f"{EDASummary.FILE_PREFIX}{EDASummary._ts()}{EDASummary.FILE_EXT}"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        flags = {
            "needs_scaling": needs_scaling,
            "high_dimensional": high_dimensional,
            "class_imbalance": class_imbalance,
            "high_collinearity": high_collinearity,
        }
        return path, summary, flags
