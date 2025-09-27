# streamlit_pages/pipeline.py
from __future__ import annotations

"""Page Pipeline: sélection d’un pipeline existant, constructeur guidé, et onglet Deep Learning.
- Bootstrap via ConfigOrchestrator (ctx complet).
- Injection de la requête pipeline après compose (pas de dict dans overrides Hydra).
- Artefacts sous outputs/<project>/streamlit/pipeline/.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.general import GeneralOrchestrator

TITLE_PIPELINES = "TITLE_PIPELINES"
TAB_EXISTING = "TAB_EXISTING"
TAB_BUILDER = "TAB_BUILDER"
TAB_DL = "TAB_DL"

LBL_SELECT_PIPELINE = "LBL_SELECT_PIPELINE"
LBL_USE_CV = "LBL_USE_CV"
LBL_CV_KIND = "LBL_CV_KIND"
LBL_FOLDS = "LBL_FOLDS"
LBL_CV_SHUFFLE = "LBL_CV_SHUFFLE"
LBL_RANDOM_STATE = "LBL_RANDOM_STATE"
BTN_RUN_SELECTED = "BTN_RUN_SELECTED"
SECTION_LATEST = "SECTION_LATEST"
LBL_LATEST_CV = "LBL_LATEST_CV"
MSG_NO_PIPELINE_CV = "MSG_NO_PIPELINE_CV"

# Builder labels
LBL_PROBLEM_TYPE = "LBL_PROBLEM_TYPE"
LBL_SCALING = "LBL_SCALING"
LBL_IMPUTATION = "LBL_IMPUTATION"
LBL_ENCODING = "LBL_ENCODING"
LBL_MODEL = "LBL_MODEL"
LBL_METRICS = "LBL_METRICS"
BTN_RUN_CUSTOM = "BTN_RUN_CUSTOM"

DIR_PIPELINES = "streamlit/pipeline"
GLOB_CV = "cv_*.csv"
HYDRA_CONFIG_NAME = "config"


def _project_root(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name


@st.cache_data
def _list_cv_results(pipes_path: Path) -> list[Path]:
    return sorted(pipes_path.glob(GLOB_CV))


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _compose_and_bootstrap(outputs_dir: str, project_name: str):
    conf_dir = (Path(__file__).resolve().parents[1] / "conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name=HYDRA_CONFIG_NAME,
            overrides=[
                f"project.output_dir={outputs_dir}",
                f"project.name={project_name}",
                f"orchestrators.pipeline.out_dir={Path(outputs_dir) / project_name / DIR_PIPELINES}",
            ],
        )
    cfg_mgr = ConfigManager(cfg)
    cfg_orch = ConfigOrchestrator(cfg_mgr)
    ctx = cfg_orch.run()
    return cfg, cfg_mgr, cfg_orch, ctx


def _run_with_request(cfg, cfg_mgr: ConfigManager, ctx: dict[str, str]) -> None:
    cfg_mgr.cfg = cfg
    cfg_mgr.load()
    with st.spinner("Running pipeline..."):
        GeneralOrchestrator(cfg_mgr, ctx=ctx).run()


def _render_latest(tr, pipes_path: Path) -> None:
    st.subheader(tr(SECTION_LATEST))
    csvs = _list_cv_results(pipes_path)
    if csvs:
        latest = csvs[-1]
        st.caption(f"{tr(LBL_LATEST_CV)}: {latest.name}")
        df = _load_csv(latest)
        st.dataframe(df, use_container_width=True)
    else:
        st.info(tr(MSG_NO_PIPELINE_CV))


def _tab_existing(tr, cfg, cfg_orch, ctx, pipes_path: Path) -> None:
    """Onglet: sélection d’un pipeline existant et exécution avec options CV."""
    app_cfg = cfg_orch.get_app_config()
    specs = getattr(app_cfg.orchestrators.pipeline, "pipeline", [])
    names = [getattr(s, "name", getattr(s, "id", "pipeline")) for s in specs]
    st.subheader(tr(LBL_SELECT_PIPELINE))
    if not names:
        st.warning("Aucun pipeline n’est défini dans la config.")
        return
    sel = st.selectbox(tr(LBL_SELECT_PIPELINE), names, index=0, key="existing_select_pipeline")

    spec = next(s for s in specs if getattr(s, "name", getattr(s, "id", "")) == sel)
    try:
        spec_dict = spec.model_dump()
    except Exception:
        spec_dict = {k: getattr(spec, k) for k in dir(spec) if not k.startswith("_")}
    with st.expander("Détails du pipeline", expanded=False):
        st.json(spec_dict)

    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        use_cv = st.checkbox(tr(LBL_USE_CV), value=True, key="existing_use_cv")
        cv_kind = (
            st.selectbox(tr(LBL_CV_KIND), ["GridSearchCV", "RandomizedSearchCV"], index=0, key="existing_cv_kind")
            if use_cv
            else None
        )
    with col2:
        folds = st.number_input(tr(LBL_FOLDS), min_value=2, max_value=20, value=5, step=1, key="existing_folds") if use_cv else None
        shuffle = st.checkbox(tr(LBL_CV_SHUFFLE), value=True, key="existing_shuffle") if use_cv else None
    with col3:
        random_state = (
            st.number_input(tr(LBL_RANDOM_STATE), value=42, step=1, key="existing_random_state") if use_cv else None
        )

    st.divider()
    if st.button(tr(BTN_RUN_SELECTED), key="existing_run_btn"):
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.orchestrators.pipeline.enabled = True
        cfg2.orchestrators.pipeline.active = [sel]
        if use_cv:
            cfg2.orchestrators.pipeline.cv = {
                "enabled": True,
                "kind": cv_kind,
                "folds": int(folds),
                "shuffle": bool(shuffle),
                "random_state": int(random_state),
            }
        else:
            cfg2.orchestrators.pipeline.cv = {"enabled": False}
        _run_with_request(cfg2, cfg_orch.get_config_manager(), ctx)
        st.cache_data.clear()
        st.success("Pipeline lancé.")
    _render_latest(tr, pipes_path)


def _tab_builder(tr, cfg, cfg_orch, ctx, pipes_path: Path) -> None:
    """Onglet: construction guidée d’un pipeline (hors deep learning)."""
    st.subheader(tr(LBL_PROBLEM_TYPE))
    typ = st.radio(tr(LBL_PROBLEM_TYPE), ["classification", "regression"], index=0, horizontal=True, key="builder_problem")

    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        enc = st.selectbox(tr(LBL_ENCODING), ["none", "onehot"], index=1, key="builder_encoding")
        imp = st.selectbox(tr(LBL_IMPUTATION), ["none", "simple", "iterative"], index=1, key="builder_imputation")
    with col2:
        scl = st.selectbox(tr(LBL_SCALING), ["none", "standard", "minmax"], index=1, key="builder_scaling")
        folds = st.number_input(tr(LBL_FOLDS), min_value=2, max_value=20, value=5, step=1, key="builder_folds")
    with col3:
        shuffle = st.checkbox(tr(LBL_CV_SHUFFLE), value=True, key="builder_shuffle")
        random_state = st.number_input(tr(LBL_RANDOM_STATE), value=42, step=1, key="builder_random_state")

    models_cls = ["logreg", "rf", "xgb", "svc", "lgbm"]
    models_reg = ["lasso", "rf", "xgb", "svr", "lgbm"]
    model = st.selectbox(tr(LBL_MODEL), models_cls if typ == "classification" else models_reg, key="builder_model")
    metrics = ["accuracy", "f1"] if typ == "classification" else ["rmse", "mae", "r2"]
    chosen_metrics = st.multiselect(tr(LBL_METRICS), options=metrics, default=metrics[:1], key="builder_metrics")

    params: dict[str, Any] = {}
    if model in ("logreg", "svc", "svr"):
        params["C"] = st.number_input("C", value=1.0, key="builder_C")
    if model in ("rf", "xgb", "lgbm"):
        params["n_estimators"] = st.number_input("n_estimators", value=200, step=50, key="builder_n_estimators")

    use_cv = st.checkbox(tr(LBL_USE_CV), value=True, key="builder_use_cv")
    cv_kind = (
        st.selectbox(tr(LBL_CV_KIND), ["GridSearchCV", "RandomizedSearchCV"], index=0, key="builder_cv_kind") if use_cv else None
    )

    st.divider()
    if st.button(tr(BTN_RUN_CUSTOM), key="builder_run_btn"):
        request = {
            "target": typ,
            "preprocessing": {"encoding": enc, "imputation": imp, "scaling": scl},
            "model": {"name": model, "params": {k: v for k, v in params.items() if v is not None}},
            "metrics": chosen_metrics,
            "cv": {
                "enabled": use_cv,
                "kind": cv_kind,
                "folds": int(folds),
                "shuffle": bool(shuffle),
                "random_state": int(random_state),
            },
        }
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.orchestrators.pipeline.enabled = True
        cfg2.orchestrators.pipeline.request = request
        cfg2.orchestrators.pipeline.out_dir = str(pipes_path)
        _run_with_request(cfg2, cfg_orch.get_config_manager(), ctx)
        st.cache_data.clear()
        st.success("Pipeline lancé.")
    _render_latest(tr, pipes_path)


def _tab_dl(tr, cfg, cfg_orch, ctx, pipes_path: Path) -> None:
    """Onglet: constructeur de réseau dense (keras) et exécution."""
    st.subheader("Dense Neural Network")
    typ = st.radio(tr(LBL_PROBLEM_TYPE), ["classification", "regression"], index=0, horizontal=True, key="dl_problem")

    st.markdown("Layers")
    layers: list[dict[str, Any]] = []
    for i in range(1, 4):
        with st.expander(f"Layer {i}", expanded=(i == 1)):
            units = st.number_input(f"units_{i}", min_value=1, max_value=2048, value=128 if i == 1 else 64, step=1, key=f"dl_units_{i}")
            act = st.selectbox(f"activation_{i}", ["relu", "tanh", "gelu"], index=0, key=f"dl_activation_{i}")
            dr = st.slider(f"dropout_{i}", min_value=0.0, max_value=0.8, value=0.2 if i == 1 else 0.0, step=0.05, key=f"dl_dropout_{i}")
            layers.append({"type": "dense", "units": int(units), "activation": act, "dropout": float(dr)})

    st.markdown("Compile")
    lr = st.number_input("learning_rate", value=1e-3, format="%.5f", key="dl_lr")
    optimizer = st.selectbox("optimizer", ["adam", "sgd", "rmsprop"], index=0, key="dl_optimizer")
    if typ == "classification":
        loss = st.selectbox("loss", ["binary_crossentropy", "categorical_crossentropy"], index=0, key="dl_loss")
        dl_metrics = st.multiselect(tr(LBL_METRICS), options=["accuracy", "f1"], default=["accuracy"], key="dl_metrics")
    else:
        loss = st.selectbox("loss", ["mse", "mae"], index=0, key="dl_loss")
        dl_metrics = st.multiselect(tr(LBL_METRICS), options=["rmse", "mae"], default=["rmse"], key="dl_metrics")

    use_cv = st.checkbox("Use CV", value=False, key="dl_use_cv")

    st.divider()
    if st.button(tr(BTN_RUN_CUSTOM) + " (DL)", key="dl_run_btn"):
        request = {
            "target": typ,
            "dl": {
                "framework": "keras",
                "layers": layers,
                "compile": {"optimizer": optimizer, "learning_rate": float(lr), "loss": loss, "metrics": dl_metrics},
            },
            "cv": {"enabled": bool(use_cv)},
        }
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.orchestrators.pipeline.enabled = True
        cfg2.orchestrators.pipeline.request = request
        cfg2.orchestrators.pipeline.out_dir = str(pipes_path / "dl")
        _run_with_request(cfg2, cfg_orch.get_config_manager(), ctx)
        st.cache_data.clear()
        st.success("Deep Learning lancé.")
    _render_latest(tr, pipes_path / "dl")


def run() -> None:
    """Affiche la page Pipeline avec 3 onglets: Existants, Constructeur, Deep Learning."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr(TITLE_PIPELINES))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = _project_root(outputs_dir, project_name)
    pipes_path = root / DIR_PIPELINES
    pipes_path.mkdir(parents=True, exist_ok=True)

    cfg, cfg_mgr, cfg_orch, ctx = _compose_and_bootstrap(outputs_dir, project_name)

    tab1, tab2, tab3 = st.tabs([tr(TAB_EXISTING), tr(TAB_BUILDER), tr(TAB_DL)])
    with tab1:
        _tab_existing(tr, cfg, cfg_orch, ctx, pipes_path)
    with tab2:
        _tab_builder(tr, cfg, cfg_orch, ctx, pipes_path)
    with tab3:
        _tab_dl(tr, cfg, cfg_orch, ctx, pipes_path)
