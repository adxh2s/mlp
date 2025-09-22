from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
)

# Typage explicite du score function pour apaiser Pylance/mypy
ScoreFunc = Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]
F_CLASSIF: ScoreFunc = cast(ScoreFunc, f_classif)


class SelectorsFactory:
    KEY_VARIANCE_THRESHOLD = "variance_threshold"
    KEY_SELECT_K_BEST = "select_k_best"
    KEY_SELECT_PERCENTILE = "select_percentile"
    PASSTHROUGH = "passthrough"

    @staticmethod
    def make_selector(cfg: dict[str, Any] | None):
        """
        Build a feature selector from a spec:
        - {"variance_threshold": 0.0}
        - {"select_k_best": 100}
        - {"select_percentile": 50}
        Returns "passthrough" when no selector is requested.
        """
        if not cfg:
            return SelectorsFactory.PASSTHROUGH

        if (
            SelectorsFactory.KEY_VARIANCE_THRESHOLD in cfg
            and cfg[SelectorsFactory.KEY_VARIANCE_THRESHOLD] is not None
        ):
            return VarianceThreshold(threshold=cfg[SelectorsFactory.KEY_VARIANCE_THRESHOLD])

        if SelectorsFactory.KEY_SELECT_K_BEST in cfg:
            return SelectKBest(score_func=F_CLASSIF, k=cfg[SelectorsFactory.KEY_SELECT_K_BEST])

        if SelectorsFactory.KEY_SELECT_PERCENTILE in cfg:
            return SelectPercentile(
                score_func=F_CLASSIF,
                percentile=cfg[SelectorsFactory.KEY_SELECT_PERCENTILE],
            )

        return SelectorsFactory.PASSTHROUGH

    @staticmethod
    def from_spec(cfg: dict[str, Any] | None):
        """Alias kept for symmetry with other factories."""
        return SelectorsFactory.make_selector(cfg)

    @staticmethod
    def instantiate_estimator(cfg: dict[str, Any] | None):
        # Backward-compat alias
        return SelectorsFactory.make_selector(cfg)
