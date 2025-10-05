from __future__ import annotations

import os
import glob
from typing import Any, Callable, MutableMapping, cast
import streamlit as st

from src.instrumentation.decorators import log_page, log_call_ex

SS_DOCS_DIR = "docs_dir"
SS_RENDER_DOCS = "render_docs"
SS_DATA_RESULT = "data_result"

def _fallback_render_docs(section: str) -> None:
    base = cast(str, st.session_state.get(SS_DOCS_DIR, "docs"))
    patterns = [os.path.join(base, f"{section}.*.md"), os.path.join(base, section, "*.md")]
    seen: set[str] = set()
    candidates: list[str] = []
    for pat in patterns:
        for path in glob.glob(pat):
            if path not in seen:
                seen.add(path)
                candidates.append(path)
    for path in sorted(candidates, key=lambda p: os.path.basename(p)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=False)
        except Exception as e:
            st.warning(f"Impossible de lire {path}: {e}")

@log_call_ex(name="home.render_docs")
def _render_docs_safe(renderer: Callable[[str], None], section: str) -> None:
    renderer(section)

@log_page("home")
def run() -> None:
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda s, **p: s))
    st.header(tr("NAV_HOME", None) if callable(tr) else "Accueil")

    renderer = cast(Callable[[str], None], st.session_state.get(SS_RENDER_DOCS, _fallback_render_docs))
    try:
        _render_docs_safe(renderer, "home")
    except Exception:
        st.warning("Rendu de documentation indisponible, affichage de secours.")
        _fallback_render_docs("home")

    data = cast(MutableMapping[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    if data:
        X = data.get("X")
        y = data.get("y")
        meta = data.get("metadata", {})
        st.subheader("Statut du dataset")
        if X is not None:
            st.caption(f"- échantillons: {getattr(X, 'shape', ['?', '?'])[0]} | features: {getattr(X, 'shape', ['?', '?'])[1]}")
            st.caption(f"- cible présente: {y is not None}")
            st.caption(f"- meta: {meta}")
            with st.expander("Aperçu (10 premières lignes)", expanded=False):
                st.dataframe(X.head(10))
        else:
            st.info("Aucun dataset disponible (orchestrateurs File/Data désactivés ou aucun fichier trouvé).")
