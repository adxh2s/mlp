from __future__ import annotations

import functools
import os
import time
import logging
from typing import Any, Callable, TypeVar, ParamSpec

# Optional dependencies
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

# Guard to check Streamlit run context to avoid CLI warnings
try:
    from streamlit.runtime.scriptrunner import get_script_run_context  # type: ignore
except Exception:  # pragma: no cover
    def get_script_run_context():
        return None  # type: ignore

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None  # type: ignore

"""
Logging decorators for pages and generic call tracing with robust fallbacks.

Priority for logger resolution inside wrappers:
1) Bound instance logger: self.get_logger(name) or self.log (LoggerMixin).
2) Streamlit session logger: st.session_state["logger_manager"], only if a Streamlit run context exists.
3) structlog.get_logger(name) if structlog is available.
4) logging.getLogger(name) as last resort.

Also provides:
- log_call_ex: emits call_start/call_end and call_error (with duration_ms and error, with stacktrace) and can attach an argument summary.
- summarize_df_y: small helper to attach X/y shapes without logging full data.
- Optional debug trace of chosen backend when MLP_DECORATORS_DEBUG=1.
"""

T = TypeVar("T")
P = ParamSpec("P")

_DEC_DEBUG = os.getenv("MLP_DECORATORS_DEBUG") == "1"
_DEC_LOGGER = logging.getLogger("decorators")


def _debug(msg: str, **fields: Any) -> None:
    if _DEC_DEBUG:
        try:
            _DEC_LOGGER.info(msg, **fields)
        except Exception:  # pragma: no cover
            pass


def _resolve_logger(args: tuple[Any, ...], label: str):
    """Best-effort logger resolution across multiple environments."""
    # 1) Instance-bound logger on self
    if args:
        self_obj = args[0]
        get_logger = getattr(self_obj, "get_logger", None)
        if callable(get_logger):
            try:
                log = get_logger(label)
                _debug("decorator_backend", selected="self.get_logger", func=label)
                return log
            except Exception:  # pragma: no cover
                pass
        bound_log = getattr(self_obj, "log", None)
        if bound_log is not None:
            _debug("decorator_backend", selected="self.log", func=label)
            return bound_log

    # 2) Streamlit session logger (only when running inside Streamlit)
    if st is not None:
        try:
            if get_script_run_context() is not None:
                logger_manager = st.session_state.get("logger_manager")  # type: ignore[attr-defined]
                if logger_manager is not None and hasattr(logger_manager, "get_logger"):
                    _debug("decorator_backend", selected="streamlit.session_state.logger_manager", func=label)
                    return logger_manager.get_logger(label)
        except Exception:  # pragma: no cover
            pass

    # 3) structlog
    if structlog is not None:
        try:
            _debug("decorator_backend", selected="structlog", func=label)
            return structlog.get_logger(label)
        except Exception:  # pragma: no cover
            pass

    # 4) stdlib logging
    _debug("decorator_backend", selected="logging", func=label)
    return logging.getLogger(label)


def log_page(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for Streamlit page entry points with start/end, duration_ms, and page_error on exceptions."""
    def deco(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            log = _resolve_logger(args, f"streamlit.page.{name}")
            try:
                log.info("page_start", page=name)
            except Exception:  # pragma: no cover
                pass

            t0 = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except Exception:
                # Emit page_error with stacktrace and duration
                dur_err = round((time.monotonic() - t0) * 1000)
                try:
                    # log.exception records exc_info=True with structlog or stdlib
                    log.exception("page_error", page=name, duration_ms=dur_err)
                except Exception:  # pragma: no cover
                    try:
                        log.error("page_error", page=name, duration_ms=dur_err, exc_info=True)
                    except Exception:  # pragma: no cover
                        pass
                raise
            finally:
                dur = round((time.monotonic() - t0) * 1000)
                try:
                    log.info("page_end", page=name, duration_ms=dur)
                except Exception:  # pragma: no cover
                    pass

        return wrapper
    return deco


def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator emitting call_start/call_end with duration_ms at INFO level."""
    def deco(fn: Callable[P, T]) -> Callable[P, T]:
        label = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "call"))

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            log = _resolve_logger(args, label)

            # Start
            try:
                log.info("call_start", func=label)
            except Exception:  # pragma: no cover
                pass

            t0 = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                dur = round((time.monotonic() - t0) * 1000)
                try:
                    log.info("call_end", func=label, duration_ms=dur)
                except Exception:  # pragma: no cover
                    pass

        return wrapper
    return deco


# ---- Argument summaries helpers ------------------------------------------------

def _safe_shape(obj: Any):
    try:
        return tuple(getattr(obj, "shape", (None, None)))
    except Exception:
        return None


def summarize_df_y(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract lightweight metadata about X/y without logging raw data:
    - Detects X and y from kwargs (X/x, y) or positional args (0/1).
    - Returns shapes, row/column counts, and whether y is present.
    """
    X = kwargs.get("X") or kwargs.get("x")
    y = kwargs.get("y")
    if X is None and len(args) >= 1:
        X = args[0]
    if y is None and len(args) >= 2:
        y = args[1]

    shp = _safe_shape(X)
    y_shp = _safe_shape(y) if y is not None else None
    n_rows = shp[0] if isinstance(shp, tuple) and len(shp) > 0 else None
    n_cols = shp[1] if isinstance(shp, tuple) and len(shp) > 1 else None
    return {
        "x_shape": shp,
        "y_shape": y_shp,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "y_present": y is not None,
    }


# ---- Extended decorator with error logging ------------------------------------

def log_call_ex(
    name: str | None = None,
    *,
    arg_summary: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]] | None = None,
    level: str = "info",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Like log_call, but also emits call_error on exceptions and can attach an argument summary.
    - call_start: always emitted at the chosen level.
    - call_end: always emitted with duration_ms at the chosen level.
    - call_error: emitted at error level with duration_ms, error message, and stacktrace, then the exception is re-raised.
    - arg_summary: optional callable receiving (args, kwargs) and returning a small dict to attach to all events.
    """
    def deco(fn: Callable[P, T]) -> Callable[P, T]:
        label = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "call"))

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            log = _resolve_logger(args, label)

            fields: dict[str, Any] = {}
            if arg_summary:
                try:
                    fields.update(arg_summary(args, kwargs) or {})
                except Exception:  # pragma: no cover
                    pass

            # Start
            try:
                getattr(log, level)("call_start", func=label, **fields)
            except Exception:  # pragma: no cover
                pass

            t0 = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                dur = round((time.monotonic() - t0) * 1000)
                try:
                    # Prefer exception() to ensure exc_info=True across backends
                    log.exception("call_error", func=label, duration_ms=dur, error=str(e), **fields)
                except Exception:  # pragma: no cover
                    try:
                        log.error("call_error", func=label, duration_ms=dur, error=str(e), exc_info=True, **fields)
                    except Exception:  # pragma: no cover
                        pass
                raise
            finally:
                dur = round((time.monotonic() - t0) * 1000)
                try:
                    getattr(log, level)("call_end", func=label, duration_ms=dur, **fields)
                except Exception:  # pragma: no cover
                    pass

        return wrapper
    return deco
