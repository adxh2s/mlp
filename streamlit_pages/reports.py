from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

APP_TITLE = "Rapports"
REPORTS_DIR = "reports"


@st.cache_resource
def get_project_root(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name


@st.cache_data
def list_artifacts(rep_path: Path, exts: List[str]) -> List[Path]:
    found: List[Path] = []
    for ext in exts:
        found.extend(rep_path.glob(f"*{ext}"))
    return sorted(found)


def run() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title("Rapports rendus")

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = get_project_root(outputs_dir, project_name)
    rep_root = root / REPORTS_DIR

    if not rep_root.exists():
        st.info("Aucun répertoire reports trouvé.")
        return

    artifacts = list_artifacts(rep_root, exts=[".html", ".md"])
    if not artifacts:
        st.info("Aucun rapport rendu trouvé.")
        return

    sel = st.selectbox("Sélectionner un rapport", options=[p.name for p in artifacts])
    f = rep_root / sel
    st.write(f"Fichier: {f}")
    if f.suffix.lower() == ".md":
        content = f.read_text(encoding="utf-8")
        st.download_button("Télécharger Markdown", data=content, file_name=f.name, mime="text/markdown")
        st.markdown(content)
    elif f.suffix.lower() == ".html":
        html = f.read_text(encoding="utf-8")
        st.download_button("Télécharger HTML", data=html, file_name=f.name, mime="text/html")
        st.components.v1.iframe(f"file://{f.resolve()}", height=800)
