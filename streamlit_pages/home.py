from __future__ import annotations

"""
Page Home Streamlit.

Rôle:
- Afficher la documentation d'accueil (Markdown) avec le renderer injecté par l'app principale.
- Présenter un bref statut du dataset préchargé (si File→Data a été exécuté au démarrage).
"""

import os
import glob
from typing import Any, Callable, MutableMapping, cast

import streamlit as st
from src.instrumentation.decorators import log_page

SS_DOCS_DIR = "docs_dir"
SS_RENDER_DOCS = "render_docs"
SS_DATA_RESULT = "data_result"


def _fallback_render_docs(section: str) -> None:
    """
    Rendu de secours des docs Markdown si aucun renderer n'a été injecté.
    Recherche basique dans docs/{section} et docs/{section}.*.md.
    """
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


@log_page("home")
def run() -> None:
    """
    Point d'entrée de la page Home.

    - Utilise render_docs injecté pour afficher la section 'home'.
    - Affiche un statut concis du dataset si présent dans la session.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda s, **p: s))
    st.header(tr("NAV_HOME") if callable(tr) else "Accueil")

    render_docs = cast(Callable[[str], None], st.session_state.get(SS_RENDER_DOCS, _fallback_render_docs))
    try:
        render_docs("home")
    except Exception as e:
        st.warning(f"Rendu de documentation indisponible: {e}")
        _fallback_render_docs("home")

    # Statut dataset (préchargé via File→Data par l'app principale)
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
