from __future__ import annotations

"""Page Rapports: listing et rendu des rapports HTML/MD générés."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def _project_root(outputs_dir: str, project_name: str) -> Path:
    """Retourne la racine du projet courant dans outputs/."""
    return Path(outputs_dir) / project_name


@st.cache_data
def _list_artifacts(rep_path: Path, exts: list[str]) -> list[Path]:
    """Liste les artefacts d'extensions données dans rep_path."""
    found: list[Path] = []
    for ext in exts:
        found.extend(rep_path.glob(f"*{ext}"))
    return sorted(found)


def run() -> None:
    """Affiche les rapports rendus sous outputs/<project>/reports."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr("TITLE_REPORTS"))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = _project_root(outputs_dir, project_name)
    rep_root = root / "reports"

    if not rep_root.exists():
        st.info(tr("MSG_NO_REPORTS_DIR"))
        return

    artifacts = _list_artifacts(rep_root, exts=[".html", ".md"])
    if not artifacts:
        st.info(tr("MSG_NO_REPORTS"))
        return

    sel = st.selectbox(tr("LBL_SELECT_REPORT"), artifacts, format_func=lambda p: p.name)
    if sel.suffix == ".md":
        st.markdown(sel.read_text(encoding="utf-8"))
    else:
        components.html(sel.read_text(encoding="utf-8"), height=800, scrolling=True)
