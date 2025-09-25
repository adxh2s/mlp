# streamlit_pages/eda.py

from __future__ import annotations

"""Page EDA: profil YData, résumé JSON et intégrations externes."""

import json
from pathlib import Path
from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components
from hydra import compose, initialize

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator


def _project_root(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name


@st.cache_data
def _latest_eda_paths(root: Path) -> Tuple[Path | None, Path | None]:
    eda_path = root / "eda"
    summary = sorted(eda_path.glob("eda_summary_*.json"))
    profile_html = sorted(eda_path.glob("profile_*.html"))
    return (summary[-1] if summary else None, profile_html[-1] if profile_html else None)


@st.cache_data
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_eda(outputs_dir: str, project_name: str) -> None:
    """Compose la config Hydra puis exécute l’EDA via l’orchestrateur général."""
    # Compose Hydra à partir de conf/config.yaml (résout `defaults` → clé orchestrators matérialisée)
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"project.output_dir={outputs_dir}",
                f"project.name={project_name}",
                # Ajouter ici des overrides optionnels si nécessaire, par ex.:
                # "orchestrators.data.input_dir=data/in",
                # "orchestrators.file.filename=dataset_V5.csv",
            ],
        )

    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()
    orch = GeneralOrchestrator(cfg_mgr)
    if hasattr(orch, "run_eda"):
        orch.run_eda()
    else:
        orch.run(steps=["eda"])


def run() -> None:
    """Affiche la page EDA avec détection de dataset, actions et artefacts récents."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr("TITLE_EDA"))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")
    root = _project_root(outputs_dir, project_name)

    # Détection dataset en data/in + uploader en alternative
    data_in = Path("data/in")
    candidates = sorted([*data_in.glob("*.csv"), *data_in.glob("*.xlsx"), *data_in.glob("*.json")])
    st.subheader(tr("LBL_DATASET"))
    if candidates:
        st.selectbox(tr("LBL_DETECTED_FILE"), candidates, format_func=lambda p: p.name, key="eda_dataset")
    else:
        up = st.file_uploader(tr("UPLOAD_DATASET"), type=["csv", "xlsx", "json"])
        if up:
            data_in.mkdir(parents=True, exist_ok=True)
            (data_in / up.name).write_bytes(up.getbuffer())
            st.success(f"{tr('MSG_FILE_SAVED')}: {up.name} → data/in")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.subheader(tr("SECTION_ACTIONS"))
        if st.button(tr("BTN_RUN_EDA")):
            # Optionnel: dériver des overrides à partir de la sélection UI si votre schéma Hydra les supporte
            _run_eda(outputs_dir, project_name)
            st.cache_data.clear()
            st.success(tr("MSG_EDA_STARTED"))

        st.subheader(tr("SECTION_EXTERNAL"))
        external_html = st.file_uploader(tr("UPLOAD_HTML"), type=["html"])
        if external_html:
            dest = root / "eda"
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / external_html.name
            target.write_bytes(external_html.getbuffer())
            st.success(f"{tr('MSG_FILE_SAVED')}: {target}")

    with c2:
        st.subheader(tr("SECTION_LATEST"))
        summary_path, profile_path = _latest_eda_paths(root)
        if summary_path and summary_path.exists():
            st.write(tr("LBL_EDA_JSON"))
            data = _load_json(summary_path)
            st.json(data)
        else:
            st.info(tr("MSG_NO_EDA_SUMMARY"))
        if profile_path and profile_path.exists():
            st.write(tr("LBL_EDA_PROFILE"))
            components.html(profile_path.read_text(encoding="utf-8"), height=800, scrolling=True)
        else:
            st.info(tr("MSG_NO_EDA_PROFILE"))
