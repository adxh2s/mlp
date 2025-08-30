from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from omegaconf import OmegaConf

# Instrumentation / configuration import depuis mlp
from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager

APP_TITLE = "mlp — Console (Programmatic Routing)"
DEFAULT_OUTPUTS = "outputs"
DEFAULT_PROJECT = "demo_project"


def _init_sidebar() -> Dict[str, Any]:
    """Initialize sidebar inputs and return resolved paths."""
    st.sidebar.header("Paramètres")
    outputs_dir = st.sidebar.text_input("Outputs dir", value=DEFAULT_OUTPUTS)
    project_name = st.sidebar.text_input("Projet", value=DEFAULT_PROJECT)
    root = Path(outputs_dir) / project_name
    st.sidebar.caption(f"Racine artefacts: {root}")
    return {"outputs_dir": outputs_dir, "project_name": project_name, "root": root}


def _init_logging_from_config() -> None:
    """Initialize logger backend from Hydra/Pydantic configuration once."""
    # Charge la configuration Hydra (conf/config.yaml) pour récupérer la section logger
    cfg = OmegaConf.load("conf/config.yaml")
    cfg_mgr = ConfigManager(cfg)
    app_cfg = cfg_mgr.load()

    # Instancie le logger manager depuis la configuration (stdlib/structlog)
    lm = build_logger_manager(app_cfg.logger)
    lm.configure()

    # Stocke dans session_state pour usages éventuels (pages, orchestrateurs déclenchés)
    st.session_state["logger_manager"] = lm
    # Log d'initialisation minimal (optionnel)
    log = lm.get_logger("mlp.streamlit")
    log.info("streamlit_init", extra={"extra_fields": {"title": APP_TITLE}})


def main() -> None:
    """Streamlit entrypoint with programmatic navigation and shared settings."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)

    # Initialize logging backend once (idempotent)
    if "logger_manager" not in st.session_state:
        _init_logging_from_config()

    # Persist params in session_state for pages consumption
    params = _init_sidebar()
    st.session_state["outputs_dir"] = params["outputs_dir"]
    st.session_state["project_name"] = params["project_name"]

    # Import programmatic pages (each exposes run())
    import streamlit_pages.home as home
    import streamlit_pages.eda as eda
    import streamlit_pages.pipelines as pipes
    import streamlit_pages.reports as reports

    # Define pages programmatically
    pg_home = st.Page(home.run, title="Accueil", icon="🏠")
    pg_eda = st.Page(eda.run, title="EDA + YData", icon="🧭")
    pg_pipes = st.Page(pipes.run, title="Pipelines", icon="🧪")
    pg_reports = st.Page(reports.run, title="Rapports", icon="📄")

    # Optionally filter availability based on artifacts presence
    pages = [pg_home, pg_eda, pg_pipes, pg_reports]

    # Run navigation
    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
