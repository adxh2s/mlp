from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectPercentile

class SelectorsFactory:
    KEY_VARIANCE_THRESHOLD = "variance_threshold"
    KEY_SELECT_K_BEST = "select_k_best"
    KEY_SELECT_PERCENTILE = "select_percentile"
    PASSTHROUGH = "passthrough"

    @staticmethod
    def make_selector(cfg: Optional[Dict[str, Any]]):
        if not cfg:
            return SelectorsFactory.PASSTHROUGH
        if SelectorsFactory.KEY_VARIANCE_THRESHOLD in cfg and cfg[SelectorsFactory.KEY_VARIANCE_THRESHOLD] is not None:
            return VarianceThreshold(threshold=cfg[SelectorsFactory.KEY_VARIANCE_THRESHOLD])
        if SelectorsFactory.KEY_SELECT_K_BEST in cfg:
            return SelectKBest(score_func=f_classif, k=cfg[SelectorsFactory.KEY_SELECT_K_BEST])
        if SelectorsFactory.KEY_SELECT_PERCENTILE in cfg:
            return SelectPercentile(score_func=f_classif, percentile=cfg[SelectorsFactory.KEY_SELECT_PERCENTILE])
        return SelectorsFactory.PASSTHROUGH
