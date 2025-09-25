# streamlit_pages/pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from hydra import compose, initialize

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator

"""Page Pipeline: builder d'options, exécution orchestrée, affichage des résultats de CV."""

# =========================
# Constantes (i18n et clés)
# =========================

# Sections et libellés (clé i18n)
TITLE_PIPELINES = "TITLE_PIPELINES"
SECTION_BUILDER = "SECTION_BUILDER"
SECTION_LATEST = "SECTION_LATEST"
LBL_TARGET = "LBL_TARGET"
LBL_SCALING = "LBL_SCALING"
LBL_IMPUTATION = "LBL_IMPUTATION"
LBL_FOLDS = "LBL_FOLDS"
LBL_CV_SHUFFLE = "LBL_CV_SHUFFLE"
LBL_RANDOM_STATE = "LBL_RANDOM_STATE"
LBL_MODELS = "LBL_MODELS"
LBL_METRICS = "LBL_METRICS"
BTN_RUN_PIPELINES = "BTN_RUN_PIPELINES"
MSG_PIPELINE_STARTED = "MSG_PIPELINE_STARTED"
LBL_LATEST_CV = "LBL_LATEST_CV"
MSG_NO_PIPELINE_CV = "MSG_NO_PIPELINE_CV"

# Dossiers/artefacts
DIR_PIPELINES = "pipelines"
GLOB_CV = "cv_*.csv"

# Hydra
HYDRA_CONF_PATH = "conf"
HYDRA_CONFIG_NAME = "config"
HYDRA_OVERRIDE_PROJECT_OUT = "project.output_dir"
HYDRA_OVERRIDE_PROJECT_NAME = "project.name"
HYDRA_OVERRIDE_PIPELINE_REQ = "orchestrators.pipeline.request"

# Valeurs par défaut UI
DEFAULT_TARGET = "classification"
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_SHUFFLE = True
DEFAULT_CV_RANDOM_STATE = 42
DEFAULT_MODELS = ["logreg", "rf", "xgb"]
DEFAULT_SCALING = "standard"
DEFAULT_IMPUTATION = "simple"
DEFAULT_SAMPLING = "none"
DEFAULT_METRICS_CLS = ["accuracy", "f1"]
DEFAULT_METRICS_REG = ["rmse", "mae", "r2"]
DEFAULT_FS_ENABLED = False
DEFAULT_FS_TOPK = 20


def _project_root(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name


@st.cache_data
def _list_cv_results(pipes_path: Path) -> list[Path]:
    """Liste les résultats de CV sous outputs/<project>/pipelines."""
    return sorted(pipes_path.glob(GLOB_CV))


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    """Charge un CSV en DataFrame."""
    return pd.read_csv(path)


@st.cache_data
def _default_pipeline_request() -> dict[str, Any]:
    """Structure de requête par défaut pour orchestrators.pipeline.request."""
    return {
        "target": DEFAULT_TARGET,
        "cv": {"folds": DEFAULT_CV_FOLDS, "shuffle": DEFAULT_CV_SHUFFLE, "random_state": DEFAULT_CV_RANDOM_STATE},
        "models": list(DEFAULT_MODELS),
        "scaling": DEFAULT_SCALING,
        "imputation": DEFAULT_IMPUTATION,
        "sampling": DEFAULT_SAMPLING,
        "metrics": list(DEFAULT_METRICS_CLS),
        "feature_selection": {"enabled": DEFAULT_FS_ENABLED, "top_k": DEFAULT_FS_TOPK},
    }


def _run_pipeline(outputs_dir: str, project_name: str, request: dict[str, Any]) -> None:
    """Compose Hydra et exécute la phase pipeline via l'orchestrateur général."""
    with initialize(version_base=None, config_path=HYDRA_CONF_PATH):
        overrides = [
            f"{HYDRA_OVERRIDE_PROJECT_OUT}={outputs_dir}",
            f"{HYDRA_OVERRIDE_PROJECT_NAME}={project_name}",
            f"{HYDRA_OVERRIDE_PIPELINE_REQ}={request}",
        ]
        cfg = compose(config_name=HYDRA_CONFIG_NAME, overrides=overrides)

    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()
    orch = GeneralOrchestrator(cfg_mgr)
    if hasattr(orch, "run_pipeline"):
        orch.run_pipeline()
    else:
        orch.run(steps=["pipeline"])


def run() -> None:
    """Affiche la page Pipeline: builder d'options, run et résultats CV."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr(TITLE_PIPELINES))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = _project_root(outputs_dir, project_name)
    pipes_path = root / DIR_PIPELINES
    pipes_path.mkdir(parents=True, exist_ok=True)

    st.subheader(tr(SECTION_BUILDER))
    req = _default_pipeline_request()

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col1:
        req["target"] = st.selectbox(tr(LBL_TARGET), ["classification", "regression"], index=0)
        req["scaling"] = st.selectbox(tr(LBL_SCALING), ["none", "standard", "minmax"], index=1)
        req["imputation"] = st.selectbox(tr(LBL_IMPUTATION), ["none", "simple", "iterative"], index=1)
    with col2:
        folds = st.number_input(tr(LBL_FOLDS), min_value=2, max_value=20, value=req["cv"]["folds"], step=1)
        req["cv"]["folds"] = int(folds)
        req["cv"]["shuffle"] = st.checkbox(tr(LBL_CV_SHUFFLE), value=req["cv"]["shuffle"])
        req["cv"]["random_state"] = st.number_input(tr(LBL_RANDOM_STATE), value=req["cv"]["random_state"], step=1)
    with col3:
        models_all = ["logreg", "rf", "xgb", "svc", "lgbm"]
        sel = st.multiselect(tr(LBL_MODELS), options=models_all, default=req["models"])
        req["models"] = sel or req["models"]
        metrics = DEFAULT_METRICS_CLS if req["target"] == "classification" else DEFAULT_METRICS_REG
        req["metrics"] = st.multiselect(tr(LBL_METRICS), options=metrics, default=req["metrics"])

    st.divider()
    if st.button(tr(BTN_RUN_PIPELINES)):
        _run_pipeline(outputs_dir, project_name, req)
        st.cache_data.clear()
        st.success(tr(MSG_PIPELINE_STARTED))

    st.subheader(tr(SECTION_LATEST))
    csvs = _list_cv_results(pipes_path)
    if csvs:
        latest = csvs[-1]
        st.caption(f"{tr(LBL_LATEST_CV)}: {latest.name}")
        df = _load_csv(latest)
        st.dataframe(df, use_container_width=True)
    else:
        st.info(tr(MSG_NO_PIPELINE_CV))
