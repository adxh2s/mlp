from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.decomposition import PCA

try:
    import umap
except Exception:
    umap = None


class ReducersFactory:
    TYPE_PCA = "pca"
    TYPE_UMAP = "umap"
    PASSTHROUGH = "passthrough"

    @staticmethod
    def make_reducer(cfg: Optional[Dict[str, Any]], random_state: int = 42):
        if not cfg:
            return ReducersFactory.PASSTHROUGH
        rtype = cfg.get("type")
        if rtype == ReducersFactory.TYPE_PCA:
            return PCA(random_state=random_state)
        if rtype == ReducersFactory.TYPE_UMAP:
            if umap is None:
                raise RuntimeError("UMAP not installed")
            return umap.UMAP(random_state=random_state)
        return ReducersFactory.PASSTHROUGH
