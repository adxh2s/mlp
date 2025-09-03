from __future__ import annotations

"""
UI constants for the Streamlit application (keys and defaults).

Keys are used for i18n lookup via the "streamlit_app" gettext domain.
Defaults provide a sane fallback if translations are missing.
"""

# i18n domain for UI texts
UI_DOMAIN = "streamlit_app"

# App identity (keys)
KEY_APP_TITLE = "app_title"

# Sidebar (keys)
KEY_SIDEBAR_SECTION = "sidebar_section"
KEY_OUTPUTS_DIR_LABEL = "outputs_dir_label"
KEY_PROJECT_LABEL = "project_label"
KEY_ROOT_PREFIX = "root_prefix"

# Page titles (keys)
KEY_PAGE_HOME = "page_home"
KEY_PAGE_EDA = "page_eda"
KEY_PAGE_PIPES = "page_pipelines"
KEY_PAGE_REPORTS = "page_reports"

# UI messages (keys)
KEY_ERR_LOAD_PAGES = "error_load_pages"
KEY_CFG_MISSING = "config_missing"
KEY_CFG_READ_ERROR = "config_read_error"

# Technical defaults (non-translated values used as fallbacks)
DEFAULT_OUTPUTS = "outputs"
DEFAULT_PROJECT = "demo_project"

# Human-readable defaults used if no .mo present or key missing
DEFAULT_TEXTS: dict[str, str] = {
    KEY_APP_TITLE: "mlp — Console (Programmatic Routing)",
    KEY_SIDEBAR_SECTION: "Paramètres",
    KEY_OUTPUTS_DIR_LABEL: "Dossier de sortie",
    KEY_PROJECT_LABEL: "Projet",
    KEY_ROOT_PREFIX: "Racine artefacts : ",
    KEY_PAGE_HOME: "Accueil",
    KEY_PAGE_EDA: "EDA + YData",
    KEY_PAGE_PIPES: "Pipelines",
    KEY_PAGE_REPORTS: "Rapports",
    KEY_ERR_LOAD_PAGES: "Erreur lors du chargement des pages : {error}",
    KEY_CFG_MISSING: "Configuration absente : {path}",
    KEY_CFG_READ_ERROR: "Erreur de lecture configuration : {error}",
}
