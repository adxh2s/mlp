from __future__ import annotations

import os
import glob
import streamlit as st

SS_DOCS_DIR: str = "docs_dir"

def _fallback_render_docs(section: str) -> None:
    base = st.session_state.get(SS_DOCS_DIR, "docs")
    patterns = [
        os.path.join(base, f"{section}.*.md"),
        os.path.join(base, section, "*.md"),
    ]
    seen = set()
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

def run() -> None:
    tr = st.session_state.get("tr", lambda s, **p: s)
    st.header(tr("NAV_HOME") if callable(tr) else "Accueil")
    # Utilise le helper global si présent, sinon le fallback local
    render_docs = st.session_state.get("render_docs", _fallback_render_docs)
    try:
        render_docs("home")
    except Exception as e:
        st.error(f"Erreur lors du rendu des documents 'home': {e}")
