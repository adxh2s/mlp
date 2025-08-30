from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator
from streamlit_pages.utils_runs import append_run, load_runs_index, new_run_id

APP_TITLE = "EDA"
EDA_DIR = "eda"


@st.cache_resource
def get_project_root(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name


@st.cache_data
def get_latest_eda_paths(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    eda_path = root / EDA_DIR
    summary = sorted(eda_path.glob("eda_summary_*.json"))
    profile_html = sorted(eda_path.glob("profile_*.html"))
    return (summary[-1] if summary else None, profile_html[-1] if profile_html else None)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_eda(outputs_dir: str, project_name: str) -> str:
    """Execute EDA phase via GeneralOrchestrator and return run_id."""
    run_id = new_run_id()
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.project.output_dir = outputs_dir
    cfg.project.name = project_name

    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()

    # Jeux de données de démo (comme dans main.py)
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"])
    y = ds.frame["target"]

    go = GeneralOrchestrator(cfg_mgr)
    out = go.run(X, y)
    root = Path(outputs_dir) / project_name
    append_run(root, {"run_id": run_id, "when": run_id, "artifacts": out.get("report", {}).get("artifacts", [])})
    return run_id


def run() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    st.title("Exploration des données (EDA)")

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = get_project_root(outputs_dir, project_name)

    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        st.subheader("Synthèse (dernier run)")
        summary_path, profile_path = get_latest_eda_paths(root)
        if summary_path and summary_path.exists():
            data = load_json(summary_path)
            st.json(data)
            st.download_button(
                "Télécharger JSON",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name=summary_path.name,
                mime="application/json",
            )
        else:
            st.info("Aucun eda_summary_*.json trouvé.")

    with colB:
        st.subheader("Profil YData (HTML intégré)")
        if profile_path and profile_path.exists():
            components.iframe(f"file://{profile_path.resolve()}", height=800)
            st.caption(f"Profil: {profile_path.name}")
        else:
            st.info("Aucun profile_*.html trouvé.")

    st.divider()
    st.subheader("Lancer un run EDA maintenant")
    if st.button("Run EDA"):
        run_id = _run_eda(outputs_dir, project_name)
        st.success(f"EDA lancé, run_id={run_id}")
        # Invalider les caches pour rafraîchir la page
        st.cache_data.clear()
        st.cache_resource.clear()
