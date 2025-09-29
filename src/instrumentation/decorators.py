from __future__ import annotations
import functools
import time
from typing import Any, Callable, TypeVar, ParamSpec
import streamlit as st

T = TypeVar("T")
P = ParamSpec("P")

def log_page(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def deco(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            lm = st.session_state.get("lm")
            log = lm.get_logger(f"streamlit.page.{name}") if lm else None
            if log: log.info("page_start", extra={"extra_fields": {"page": name}})
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                dur = round((time.time() - t0) * 1000)
                if log: log.info("page_end", extra={"extra_fields": {"page": name, "duration_ms": dur}})
        return wrapper
    return deco

def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def deco(fn: Callable[P, T]) -> Callable[P, T]:
        label = name or fn.__name__
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            lm = st.session_state.get("lm")
            log = lm.get_logger(f"streamlit.fn.{label}") if lm else None
            if log: log.debug("call_start", extra={"extra_fields": {"fn": label}})
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                dur = round((time.time() - t0) * 1000)
                if log: log.debug("call_end", extra={"extra_fields": {"fn": label, "duration_ms": dur}})
        return wrapper
    return deco
