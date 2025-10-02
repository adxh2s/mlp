from __future__ import annotations

import os, re
from collections.abc import Callable
import streamlit as st

from src.instrumentation.decorators import log_page

DEFAULT_LOG_FILE = os.getenv("MLP_LOG_FILE", "streamlit_app.log")

def _read_tail(path: str, max_lines: int) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-max_lines:] if max_lines > 0 else lines
    except Exception as e:
        return [f"[logs] Impossible de lire '{path}': {e}\n"]

@log_page("logs")
def run() -> None:
    tr: Callable[[str], str] = st.session_state.get("tr", lambda s, **p: s)
    st.header(tr("NAV_LOGS") if callable(tr) else "Logs")

    log_file = st.text_input("Fichier log", value=st.session_state.get("log_file", DEFAULT_LOG_FILE))
    st.session_state["log_file"] = log_file

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_lines = st.number_input("Dernières lignes", min_value=50, max_value=10000, step=50, value=500)
    with col2:
        pattern = st.text_input("Filtre (regex)", value="")
    with col3:
        if st.button("Rafraîchir"):
            pass

    lines = _read_tail(log_file, int(max_lines))
    if pattern:
        try:
            rx = re.compile(pattern)
            lines = [ln for ln in lines if rx.search(ln)]
        except re.error as e:
            st.warning(f"Regex invalide: {e}")

    st.subheader(os.path.basename(log_file))
    st.text("".join(lines))
