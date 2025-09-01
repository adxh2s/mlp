from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator
from streamlit_pages.utils_runs import append_run, new_run_id

APP_TITLE = "Pipelines"
PIPELINES_DIR = "pipelines"


@st.cache_resource
def get_project_root(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name


@st.cache_data
def list_cv_results(pipes_path: Path) -> List[Path]:
    return sorted(pipes_path.glob("cv_*.csv"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _run_pipelines(outputs_dir: str, project_name: str) -> str:
    run_id = new_run_id()
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.project.output_dir = outputs_dir
    cfg.project.name = project_name

    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()

    # Dataset de démo
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"])
    y = ds.frame["target"]

    go = GeneralOrchestrator(cfg_mgr)
    out = go.run(X, y)
    root = Path(outputs_dir) / project_name
    append_run(
        root,
        {"run_id": run_id, "when": run_id, "artifacts": out.get("report", {}).get("artifacts", [])},
    )
    return run_id


def run() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
    st.title("Pipelines")

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = get_project_root(outputs_dir, project_name)
    pipes_root = root / PIPELINES_DIR

    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        if not pipes_root.exists():
            st.info("Aucun répertoire pipelines trouvé.")
        else:
            csvs = list_cv_results(pipes_root)
            if not csvs:
                st.info("Aucun cv_*.csv trouvé.")
            else:
                sel = st.selectbox("Résultats CV", options=[p.name for p in csvs])
                selected = pipes_root / sel
                df = load_csv(selected)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Télécharger CSV",
                    data=df.to_csv(index=False),
                    file_name=selected.name,
                    mime="text/csv",
                )

    with colB:
        st.subheader("Lancer un run Pipelines")
        if st.button("Run Pipelines"):
            run_id = _run_pipelines(outputs_dir, project_name)
            st.success(f"Pipelines lancés, run_id={run_id}")
            st.cache_data.clear()
            st.cache_resource.clear()
