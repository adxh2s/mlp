from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA

# Import optionnel d'UMAP avec nom de classe isolé pour contenter mypy/pylance
try:
    import umap  # type: ignore

    UMAP_CLS = umap.UMAP
except Exception:  # pragma: no cover
    UMAP_CLS = None  # type: ignore[assignment]


class ReducersFactory:
    TYPE_PCA = "pca"
    TYPE_UMAP = "umap"
    PASSTHROUGH = "passthrough"

    @staticmethod
    def make_reducer(cfg: dict[str, Any] | None, random_state: int = 42):
        """
        Build a dimensionality reduction transformer from a spec:
        - {"type": "pca", "params": {...}}
        - {"type": "umap", "params": {...}}
        Returns "passthrough" when no reduction is requested.
        """
        if not cfg:
            return ReducersFactory.PASSTHROUGH

        rtype = cfg.get("type")

        # Assurer un dict typé pour apaiser l'analyse statique
        params_obj = cfg.get("params")
        params_cfg: dict[str, Any] = params_obj if isinstance(params_obj, dict) else {}

        # Ne passer que les scalaires à l'estimateur de base; garder les listes pour la grille
        base_params: dict[str, Any] = {}
        for k, v in params_cfg.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                base_params[k] = v

        if rtype == ReducersFactory.TYPE_PCA:
            # Pertinent si svd_solver="randomized"; inoffensif sinon sur sklearn récent
            base_params.setdefault("random_state", random_state)
            return PCA(**base_params)

        if rtype == ReducersFactory.TYPE_UMAP:
            if UMAP_CLS is None:
                raise RuntimeError("UMAP not installed")
            base_params.setdefault("random_state", random_state)
            return UMAP_CLS(**base_params)

        return ReducersFactory.PASSTHROUGH

    @staticmethod
    def from_spec(cfg: dict[str, Any] | None, random_state: int = 42):
        """Alias kept for backward compatibility."""
        return ReducersFactory.make_reducer(cfg, random_state)

    @staticmethod
    def instantiate_estimator(cfg: dict[str, Any] | None, random_state: int = 42):
        return ReducersFactory.make_reducer(cfg, random_state)
