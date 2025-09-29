from __future__ import annotations

"""
Page Logs Streamlit.

Rôle:
- Afficher la fin d'un fichier de log, avec un filtre regex et rafraîchissement manuel.
- Valeur par défaut: outputs_root/streamlit_app.log si non fournie.
"""

import os
import re
from typing import Callable, List, cast

import streamlit as st
from src.instrumentation.decorators import log_page

SS_CTX = "ctx"
DEFAULT_LOG_FILE_ENV = os.getenv("MLP_LOG_FILE", "streamlit_app.log")


def _read_tail(path: str, max_lines: int) -> List[str]:
    """Lit les dernières lignes d'un fichier texte."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-max_lines:] if max_lines > 0 else lines
    except Exception as e:  # noqa: BLE001
        return [f"[logs] Impossible de lire '{path}': {e}\n"]


@log_page("logs")
def run() -> None:
    """
    Point d'entrée de la page Logs.

    - Sélection du fichier de log à lire.
    - Affichage avec filtre regex optionnel.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda s, **p: s))
    st.header("Logs")

    default_log = DEFAULT_LOG_FILE_ENV
    try:
        outputs_root = cast(str, st.session_state.get(SS_CTX, {}).get("outputs_root", ""))
        if outputs_root:
            default_log = os.path.join(outputs_root, "streamlit_app.log")
    except Exception:
        pass

    log_file = st.text_input("Fichier log", value=st.session_state.get("log_file", default_log))
    st.session_state["log_file"] = log_file

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_lines = st.number_input("Dernières lignes", min_value=50, max_value=10000, step=50, value=500)
    with col2:
        pattern = st.text_input("Filtre (regex)", value=st.session_state.get("log_filter", ""))
        st.session_state["log_filter"] = pattern
    with col3:
        if st.button("Rafraîchir"):
            st.experimental_rerun()

    lines = _read_tail(log_file, int(max_lines))
    if pattern:
        try:
            rx = re.compile(pattern)
            lines = [ln for ln in lines if rx.search(ln)]
        except Exception as e:  # noqa: BLE001
            st.warning(f"Regex invalide: {e}")

    st.text_area("Sortie", value="".join(lines), height=500)
