from __future__ import annotations

"""
Page Notebooks Streamlit.

Rôle:
- Lister et afficher les notebooks exportés en HTML.
- Fournir des liens vers les .ipynb si présents.
"""

from pathlib import Path
from typing import Callable, Tuple, cast

import streamlit as st
import streamlit.components.v1 as components

from src.instrumentation.decorators import log_page

SS_CTX = "context"


def _root(notebook_dir: str) -> Path:
    """Retourne le répertoire des notebooks à lister."""
    return Path(notebook_dir)


@st.cache_data
def _list_assets(root: Path) -> Tuple[list[Path], list[Path]]:
    """Liste tous les fichiers .html et .ipynb sous le répertoire donné."""
    htmls = sorted(root.rglob("*.html"))
    ipynb = sorted(root.rglob("*.ipynb"))
    return htmls, ipynb


@log_page("notebook")
def run() -> None:
    """
    Point d'entrée de la page Notebooks.

    - Détecte le dossier notebooks sous project_dir par défaut.
    - Affiche un viewer pour les exports HTML, et des liens vers les .ipynb.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_NOTEBOOK", None))

    # Par défaut: {project_dir}/notebooks
    project_dir = cast(str, st.session_state.get(SS_CTX, {}).get("project_dir", "."))
    notebook_dir = cast(str, st.session_state.get("notebooks_dir", str(Path(project_dir) / "notebooks")))
    notebook_url = cast(str, st.session_state.get("notebooks_url", ""))

    root = _root(notebook_dir)
    if not root.exists():
        st.info(f"{tr('MSG_NO_NOTEBOOK_DIR') if callable(tr) else 'Dossier notebooks introuvable'}: {root}")
        return

    htmls, ipynb = _list_assets(root)

    st.subheader(tr("LBL_HTML_EXPORTED", None) if callable(tr) else "Exports HTML")
    if htmls:
        sel_html = st.selectbox(
            tr("LBL_HTML_EXPORTED", None) if callable(tr) else "HTML disponibles",
            htmls,
            format_func=lambda p: p.relative_to(root),
        )
        components.html(sel_html.read_text(encoding="utf-8"), height=800, scrolling=True)
    else:
        st.caption("Aucun export HTML trouvé.")

    st.subheader(tr("LBL_NOTEBOOK", None) if callable(tr) else "Notebooks")
    if ipynb:
        for nb in ipynb:
            if notebook_url:
                st.markdown(f"- [{nb.name}]({notebook_url.rstrip('/')}/{nb.name})")
            else:
                st.markdown(f"- {nb.relative_to(root)}")
    else:
        st.caption("Aucun .ipynb trouvé.")
