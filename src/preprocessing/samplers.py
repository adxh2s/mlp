from __future__ import annotations

"""SamplersFactory: imbalanced-learn resamplers with structured telemetry."""

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

from typing import Any

# Imports optionnels (évite la redéfinition de constantes en majuscules)
try:
    from imblearn.over_sampling import SMOTE as SmoteCls  # type: ignore
    from imblearn.over_sampling import RandomOverSampler as RandomOverSamplerCls
    from imblearn.under_sampling import RandomUnderSampler as RandomUnderSamplerCls  # type: ignore
except Exception:  # pragma: no cover
    SmoteCls = None
    RandomOverSamplerCls = None
    RandomUnderSamplerCls = None


class SamplersFactory:
    """Factory for building resamplers (SMOTE/Over/Under) from specs."""

    TYPE_SMOTE = "smote"
    TYPE_UNDER = "under"
    TYPE_OVER = "over"

    PASSTHROUGH = "passthrough"

    @staticmethod
    @log_call("samplers.make_sampler")
    def make_sampler(cfg: dict[str, Any] | None):
        """
        Build a resampler from a spec (imbalanced-learn):
        - {"type": "smote", "params": {...}}
        - {"type": "over", "params": {...}} # RandomOverSampler
        - {"type": "under", "params": {...}} # RandomUnderSampler
        Returns "passthrough" when no resampling is requested.
        """
        if not cfg:
            return SamplersFactory.PASSTHROUGH

        stype = cfg.get("type")
        params: dict[str, Any] = cfg.get("params") or {}

        if stype == SamplersFactory.TYPE_SMOTE:
            if SmoteCls is None:
                raise RuntimeError("imbalanced-learn SMOTE not installed")
            return SmoteCls(**params)

        if stype == SamplersFactory.TYPE_OVER:
            if RandomOverSamplerCls is None:
                raise RuntimeError("imbalanced-learn not installed")
            return RandomOverSamplerCls(**params)

        if stype == SamplersFactory.TYPE_UNDER:
            if RandomUnderSamplerCls is None:
                raise RuntimeError("imbalanced-learn not installed")
            return RandomUnderSamplerCls(**params)

        return SamplersFactory.PASSTHROUGH

    @staticmethod
    @log_call("samplers.from_spec")
    def from_spec(cfg: dict[str, Any] | None):
        """Alias for API symmetry."""
        return SamplersFactory.make_sampler(cfg)
