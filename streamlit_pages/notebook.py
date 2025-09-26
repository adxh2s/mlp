from __future__ import annotations

"""Page Notebooks: exports HTML et liens/iframes vers serveurs Jupyter/Voilà."""

from pathlib import Path
from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components


def _root(notebook_dir: str) -> Path:
    """Retourne la racine des notebook."""
    return Path(notebook_dir)


@st.cache_data
def _list_assets(root: Path) -> Tuple[list[Path], list[Path]]:
    """Retourne la liste des HTML exportés et des fichiers .ipynb."""
    htmls = sorted(root.rglob("*.html"))
    ipynb = sorted(root.rglob("*.ipynb"))
    return htmls, ipynb


def run() -> None:
    """Affiche les exports HTML et propose l’embed d’un notebook servi par Voilà."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr("TITLE_NOTEBOOK"))

    notebook_dir = st.session_state.get("notebook_dir", "notebook")
    notebook_url = st.session_state.get("notebook_url", "")

    root = _root(notebook_dir)
    if not root.exists():
        st.info(f"{tr('MSG_NO_NOTEBOOK_DIR')}: {root}")
        return

    htmls, ipynb = _list_assets(root)

    st.subheader(tr("LBL_HTML_EXPORTED"))
    if htmls:
        sel_html = st.selectbox(tr("LBL_HTML_EXPORTED"), htmls, format_func=lambda p: p.relative_to(root))
        components.html(sel_html.read_text(encoding="utf-8"), height=800, scrolling=True)
    else:
        st.info(tr("MSG_NO_HTML"))

    st.subheader(tr("LBL_NOTEBOOK_SOURCES"))
    if ipynb:
        nb = st.selectbox("Notebook", ipynb, format_func=lambda p: p.relative_to(root))
        if notebook_url:
            # Hypothèse: Voilà sert /voila/render/
            rel = nb.relative_to(root).as_posix()
            components.iframe(f"{notebook_url.rstrip('/')}/voila/render/{rel}", height=800)
        else:
            st.markdown(f"- {nb.relative_to(root)}")
    else:
        st.info(tr("MSG_NO_NOTEBOOK"))
