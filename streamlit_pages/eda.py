# streamlit_pages/eda.py
from __future__ import annotations

"""
Page EDA: sélection/chargement du dataset en en-tête, exécution EDA, et affichage plein écran des artefacts.

Principes:
- Bootstrap via ConfigOrchestrator pour charger/valider la config et construire un ctx stable (répertoires garantis).
- Désactivation explicite de orchestrators.pipeline.enabled pendant le run EDA pour empêcher l'enchaînement vers la pipeline.
- Artefacts attendus: JSON résumé et profil HTML sous outputs/<project>/eda.
"""

import json
from pathlib import Path
from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.general import GeneralOrchestrator


TITLE_EDA = "TITLE_EDA"
LBL_DATASET = "LBL_DATASET"
LBL_DETECTED_FILE = "LBL_DETECTED_FILE"
UPLOAD_DATASET = "UPLOAD_DATASET"
MSG_FILE_SAVED = "MSG_FILE_SAVED"
BTN_RUN_EDA = "BTN_RUN_EDA"
MSG_EDA_STARTED = "MSG_EDA_STARTED"
LBL_EDA_JSON = "LBL_EDA_JSON"
LBL_EDA_PROFILE = "LBL_EDA_PROFILE"
MSG_NO_EDA_SUMMARY = "MSG_NO_EDA_SUMMARY"
MSG_NO_EDA_PROFILE = "MSG_NO_EDA_PROFILE"

HYDRA_CONFIG_NAME = "config"


def _project_root(outputs_dir: str, project_name: str) -> Path:
    """Retourne la racine du projet courant dans outputs/."""
    return Path(outputs_dir) / project_name


@st.cache_data
def _latest_eda_paths(root: Path) -> Tuple[Path | None, Path | None]:
    """Retourne les chemins des derniers artefacts EDA (summary JSON et profil HTML)."""
    eda_path = root / "eda"
    summary = sorted(eda_path.glob("eda_summary_*.json"))
    profile_html = sorted(eda_path.glob("profile_*.html"))
    return (summary[-1] if summary else None, profile_html[-1] if profile_html else None)


@st.cache_data
def _load_json(path: Path) -> dict:
    """Charge un fichier JSON en dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def _compose_and_bootstrap(outputs_dir: str, project_name: str):
    """
    Compose la config Hydra (pipeline désactivée), initialise ConfigOrchestrator, et renvoie (cfg, cfg_mgr, cfg_orch, ctx).

    Note: orchestrators.pipeline.enabled=false évite l’exécution de la pipeline lors de l’EDA.
    """
    conf_dir = (Path(__file__).resolve().parents[1] / "conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name=HYDRA_CONFIG_NAME,
            overrides=[
                f"project.output_dir={outputs_dir}",
                f"project.name={project_name}",
                "orchestrators.pipeline.enabled=false",
            ],
        )
    cfg_mgr = ConfigManager(cfg)
    cfg_orch = ConfigOrchestrator(cfg_mgr)
    ctx = cfg_orch.run()
    return cfg, cfg_mgr, cfg_orch, ctx


def _run_eda(outputs_dir: str, project_name: str) -> None:
    """
    Exécute l’EDA uniquement:
    - Compose config avec pipeline désactivée.
    - Bootstrap via ConfigOrchestrator pour obtenir un ctx stable.
    - Lance GeneralOrchestrator.run() pour générer les artefacts EDA.
    """
    cfg, cfg_mgr, cfg_orch, ctx = _compose_and_bootstrap(outputs_dir, project_name)
    # S’assure que la pipeline reste désactivée même après transformation OmegaConf éventuelle
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg2.orchestrators.pipeline.enabled = False
    cfg_mgr.cfg = cfg2
    cfg_mgr.load()
    with st.spinner("Running EDA..."):
        GeneralOrchestrator(cfg_mgr, ctx=ctx).run()


def run() -> None:
    """
    Affiche la page EDA:
    - En-tête: sélection ou upload du fichier (data/in) + bouton de lancement.
    - Corps: rendu plein écran des derniers artefacts EDA (JSON + profil HTML).
    """
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr(TITLE_EDA))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = _project_root(outputs_dir, project_name)

    # En-tête (contrôles)
    st.subheader(tr(LBL_DATASET))
    data_in = Path("data/in")
    candidates = sorted([*data_in.glob("*.csv"), *data_in.glob("*.xlsx"), *data_in.glob("*.json")])
    if candidates:
        st.selectbox(tr(LBL_DETECTED_FILE), candidates, format_func=lambda p: p.name, key="eda_dataset")
    else:
        up = st.file_uploader(tr(UPLOAD_DATASET), type=["csv", "xlsx", "json"])
        if up:
            data_in.mkdir(parents=True, exist_ok=True)
            (data_in / up.name).write_bytes(up.getbuffer())
            st.success(f"{tr(MSG_FILE_SAVED)}: {up.name} → data/in")

    if st.button(tr(BTN_RUN_EDA)):
        _run_eda(outputs_dir, project_name)
        st.cache_data.clear()
        st.success(tr(MSG_EDA_STARTED))

    st.divider()

    # Résultats (plein écran)
    summary_path, profile_path = _latest_eda_paths(root)
    if summary_path and summary_path.exists():
        st.write(tr(LBL_EDA_JSON))
        data = _load_json(summary_path)
        st.json(data, expanded=False)
    else:
        st.info(tr(MSG_NO_EDA_SUMMARY))

    if profile_path and profile_path.exists():
        st.write(tr(LBL_EDA_PROFILE))
        components.html(profile_path.read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.info(tr(MSG_NO_EDA_PROFILE))
