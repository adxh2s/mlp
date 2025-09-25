from __future__ import annotations

"""Page Pipelines: builder d'options, exécution orchestrée, et affichage résultats CV."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator


def _project_root(outputs_dir: str, project_name: str) -> Path:
    """Retourne la racine du projet courant dans outputs/."""
    return Path(outputs_dir) / project_name


@st.cache_data
def _list_cv_results(pipes_path: Path) -> list[Path]:
    """Liste les résultats de CV sous outputs/<project>/pipelines."""
    return sorted(pipes_path.glob("cv_*.csv"))


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    """Charge un CSV en DataFrame."""
    return pd.read_csv(path)


@st.cache_data
def _load_pipeline_options() -> dict:
    """Charge les options UI depuis le YAML si présent, sinon valeurs par défaut."""
    cfg_path = Path("conf/orchestrators/pipelines/pipelines.yaml")
    if cfg_path.exists():
        cfg = OmegaConf.load(str(cfg_path))
        return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[no-any-return]
    return {
        "preprocessing": {
            "encoder": ["onehot", "target", "none"],
            "imputer": ["simple", "iterative", "none"],
            "selector": ["kbest", "percentile", "none"],
            "reducer": ["none", "pca", "umap"],
        },
        "estimators": ["logreg", "rf", "xgb", "dl"],
        "scoring": ["accuracy", "f1", "roc_auc"],
        "cv": [3, 5, 10],
    }


def _run_pipelines(outputs_dir: str, project_name: str, request: dict[str, Any]) -> None:
    """Lance les pipelines via l'orchestrateur en injectant la requête UI dans la config."""
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.project.output_dir = outputs_dir
    cfg.project.name = project_name
    if "orchestrators" not in cfg:
        cfg.orchestrators = {}
    cfg.orchestrators["pipelines_request"] = request
    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()
    orch = GeneralOrchestrator(cfg_mgr)
    if hasattr(orch, "run_pipelines"):
        orch.run_pipelines()
    elif hasattr(orch, "run"):
        orch.run(steps=["pipelines"])


def run() -> None:
    """Affiche le builder de pipelines et les résultats CV disponibles."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr("TITLE_PIPELINES"))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = _project_root(outputs_dir, project_name)
    pipes_root = root / "pipelines"
    pipes_root.mkdir(parents=True, exist_ok=True)

    opts = _load_pipeline_options()
    pre = opts.get("preprocessing", {}) if isinstance(opts, dict) else {}

    with st.expander(tr("SECTION_BUILDER"), expanded=True):
        with st.form("pipeline_builder"):
            enc = st.selectbox(tr("LBL_ENCODER"), pre.get("encoder", ["onehot", "target", "none"]))
            imp = st.selectbox(tr("LBL_IMPUTER"), pre.get("imputer", ["simple", "iterative", "none"]))
            sel = st.selectbox(tr("LBL_SELECTOR"), pre.get("selector", ["kbest", "percentile", "none"]))
            red = st.selectbox(tr("LBL_REDUCER"), pre.get("reducer", ["none", "pca", "umap"]))
            est = st.selectbox(tr("LBL_ESTIMATOR"), opts.get("estimators", ["logreg", "rf", "xgb", "dl"]))
            hp = st.text_area(tr("LBL_HPARAMS"), value="{}")
            cv = st.selectbox(tr("LBL_CV"), opts.get("cv", [3, 5, 10]))
            scoring = st.selectbox(tr("LBL_SCORING"), opts.get("scoring", ["accuracy", "f1", "roc_auc"]))
            submitted = st.form_submit_button(tr("BTN_RUN_PIPELINES"))
        if submitted:
            try:
                hp_dict = dict(OmegaConf.to_container(OmegaConf.create(hp), resolve=True))  # type: ignore[assignment]
            except Exception:
                hp_dict = {}
            request = {
                "preprocessing": {"encoder": enc, "imputer": imp, "selector": sel, "reducer": red},
                "estimator": est,
                "hyperparams": hp_dict,
                "cv": int(cv),
                "scoring": str(scoring),
            }
            _run_pipelines(outputs_dir, project_name, request)
            st.cache_data.clear()
            st.success(tr("MSG_PIPELINE_STARTED"))

    st.subheader(tr("LBL_CV_RESULTS"))
    artifacts = _list_cv_results(pipes_root)
    if not artifacts:
        st.info(tr("MSG_NO_CV"))
        return

    sel_path = st.selectbox(tr("LBL_RESULTS_FILE"), artifacts, format_func=lambda p: p.name)
    df = _load_csv(sel_path)
    st.dataframe(df, use_container_width=True)
