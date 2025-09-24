# streamlit_pages/notebooks.py
from __future__ import annotations

"""Page Notebooks: intégration d'exports HTML et liens vers serveurs Jupyter."""

from pathlib import Path
from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components


def _root(notebooks_dir: str) -> Path:
    return Path(notebooks_dir)


@st.cache_data
def _list_assets(root: Path) -> Tuple[list[Path], list[Path]]:
    htmls = sorted(root.rglob("*.html"))
    ipynb = sorted(root.rglob("*.ipynb"))
    return htmls, ipynb


def run() -> None:
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.set_page_config(page_title=tr("TITLE_NOTEBOOKS"), layout="wide")
    st.title(tr("TITLE_NOTEBOOKS"))

    notebooks_dir = st.session_state.get("notebooks_dir", "notebooks")
    notebooks_url = st.session_state.get("notebooks_url", "")

    root = _root(notebooks_dir)
    if not root.exists():
        st.info(f"{tr('MSG_NO_NOTEBOOKS_DIR')}: {root}")
        return

    htmls, ipynb = _list_assets(root)

    st.subheader(tr("LBL_HTML_EXPORTED"))
    if htmls:
        sel_html = st.selectbox(tr("LBL_HTML_EXPORTED"), htmls, format_func=lambda p: p.relative_to(root))
        components.html(sel_html.read_text(encoding="utf-8"), height=800, scrolling=True)
    else:
        st.info(tr("MSG_NO_HTML"))

    st.subheader(tr("LBL_NOTEBOOKS_SOURCES"))
    if ipynb:
        for p in ipynb:
            label = str(p.relative_to(root))
            st.markdown(f"- [{label}]({notebooks_url})" if notebooks_url else f"- {label}")
    else:
        st.info(tr("MSG_NO_NOTEBOOKS"))
