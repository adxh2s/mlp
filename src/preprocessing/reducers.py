from __future__ import annotations

"""ReducersFactory: dimensionality reduction builders with structured telemetry."""

# Décorateurs: import robuste avec fallback no-op
try:
    from src.instrumentation.decorators import log_call
except Exception:  # pragma: no cover
    from typing import Callable, TypeVar, ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

    def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:  # type: ignore[override]
        def deco(fn: Callable[P, T]) -> Callable[P, T]:
            return fn
        return deco

from typing import Any

from sklearn.decomposition import PCA


class ReducersFactory:
    """Factory for building reducers (PCA/UMAP/ParametricUMAP) from specs."""

    TYPE_PCA = "pca"
    TYPE_UMAP = "umap"
    TYPE_PARAM_UMAP = "parametric_umap"  # alias interne accepté ci-dessous

    PASSTHROUGH = "passthrough"

    @staticmethod
    @log_call("reducers._get_umap")
    def _get_umap():
        """
        Import paresseux de umap-learn uniquement lorsque requis.
        Évite d'importer umap au niveau module car umap expose ParametricUMAP
        dans son __init__, ce qui peut tracter TensorFlow.
        """
        try:
            import importlib

            return importlib.import_module("umap")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "UMAP indisponible; installez umap-learn>=0.5,<0.6 ou désactivez la réduction UMAP."
            ) from e

    @staticmethod
    @log_call("reducers.make_reducer")
    def make_reducer(cfg: dict[str, Any] | None, random_state: int = 42):
        """
        Construit un réducteur dimensionnel depuis une spec:
        - {"type": "pca", "params": {...}}
        - {"type": "umap", "params": {...}}
        - {"type": "parametric_umap", "params": {...}} # si TensorFlow dispo
        Retourne "passthrough" si aucune réduction n'est demandée.
        """
        if not cfg:
            return ReducersFactory.PASSTHROUGH

        rtype = cfg.get("type")
        params_obj = cfg.get("params")
        params_cfg: dict[str, Any] = params_obj if isinstance(params_obj, dict) else {}

        # Ne passer que des scalaires à l'estimateur (les listes sont pour la grille de CV)
        base_params: dict[str, Any] = {
            k: v
            for k, v in params_cfg.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }

        if rtype == ReducersFactory.TYPE_PCA:
            # pertinent surtout si svd_solver="randomized"
            base_params.setdefault("random_state", random_state)
            return PCA(**base_params)

        if rtype == ReducersFactory.TYPE_UMAP:
            umap = ReducersFactory._get_umap()
            base_params.setdefault("random_state", random_state)
            return umap.UMAP(**base_params)

        if rtype in (ReducersFactory.TYPE_PARAM_UMAP, "pumap"):
            umap = ReducersFactory._get_umap()
            base_params.setdefault("random_state", random_state)
            try:
                return umap.ParametricUMAP(**base_params)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "ParametricUMAP requiert un TensorFlow compatible; vérifiez requirements et installation."
                ) from e

        return ReducersFactory.PASSTHROUGH

    @staticmethod
    @log_call("reducers.from_spec")
    def from_spec(cfg: dict[str, Any] | None, random_state: int = 42):
        """Alias rétro-compatible."""
        return ReducersFactory.make_reducer(cfg, random_state)

    @staticmethod
    @log_call("reducers.instantiate_estimator")
    def instantiate_estimator(cfg: dict[str, Any] | None, random_state: int = 42):
        """Alias rétro-compatible."""
        return ReducersFactory.make_reducer(cfg, random_state)
