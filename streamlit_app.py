from __future__ import annotations

import streamlit as st
# Configuration de la page (unique)
APP_TITLE: str = "MLP App"
APP_ICON: str = "📊"
PAGE_LAYOUT: str = "wide"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)

import os
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.message import MessageOrchestrator
from streamlit_pages import demo, eda, home, notebook, pipeline, report

"""Entrée principale Streamlit: i18n via gettext, navigation multipages, config robuste.

- set_page_config appelé une seule fois, après les imports (conforme Ruff).
- i18n centralisée (MessageOrchestrator) → st.session_state['tr'].
- Tolérance à l’absence de conf/config.yaml (config minimale).
- Types explicites et casts pour Pylance/mypy.
"""

# Constantes de navigation et langues
NAV_KEYS: list[str] = ["NAV_HOME", "NAV_EDA", "NAV_PIPELINE", "NAV_REPORT", "NAV_NOTEBOOK", "NAV_DEMO"]
SUPPORTED_LANGS: list[str] = ["fr", "en"]

# Clés session_state
SS_OUTPUTS_DIR: str = "outputs_dir"
SS_PROJECT_NAME: str = "project_name"
SS_NOTEBOOK_DIR: str = "notebook_dir"
SS_NOTEBOOK_URL: str = "notebook_url"
SS_LANG: str = "lang"
SS_TR: str = "tr"
SS_NAV_INDEX: str = "nav_index"

# Env vars (défauts)
ENV_OUTPUTS_DIR: str = "MLP_OUTPUTS_DIR"
ENV_PROJECT_NAME: str = "MLP_PROJECT_NAME"
ENV_NOTEBOOK_DIR: str = "MLP_NOTEBOOK_DIR"
ENV_NOTEBOOK_URL: str = "MLP_NOTEBOOK_URL"
ENV_LANG: str = "MLP_LANG"

# i18n / orchestrateur
I18N_DOMAIN: str = "streamlit_app"
I18N_LOCALES_DIR_DEFAULT: str = "i18n/locales"

# Type alias
Translator = Callable[[str], str]


def _init_defaults() -> None:
    """Initialise les paramètres globaux de session si absents."""
    ss: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)
    defaults: dict[str, Any] = {
        SS_OUTPUTS_DIR: os.getenv(ENV_OUTPUTS_DIR, "outputs"),
        SS_PROJECT_NAME: os.getenv(ENV_PROJECT_NAME, "demo_project"),
        SS_NOTEBOOK_DIR: os.getenv(ENV_NOTEBOOK_DIR, "notebook"),
        SS_NOTEBOOK_URL: os.getenv(ENV_NOTEBOOK_URL, ""),
        SS_LANG: os.getenv(ENV_LANG, "fr"),
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v


def _load_or_create_cfg(lang: str) -> DictConfig:
    """Charge conf/config.yaml ou crée une config minimale si absent."""
    try:
        cfg = OmegaConf.load("conf/config.yaml")
    except Exception:
        cfg = OmegaConf.create(
            {
                "project": {
                    "output_dir": st.session_state.get(SS_OUTPUTS_DIR, "outputs"),
                    "name": st.session_state.get(SS_PROJECT_NAME, "demo_project"),
                },
                "orchestrators": {"message": {"locale": lang, "locales_dir": I18N_LOCALES_DIR_DEFAULT}},
            }
        )
    if "orchestrators" not in cfg:
        cfg.orchestrators = {}
    if "message" not in cfg.orchestrators:
        cfg.orchestrators.message = {}
    cfg.orchestrators.message.locale = lang
    if "locales_dir" not in cfg.orchestrators.message:
        cfg.orchestrators.message.locales_dir = I18N_LOCALES_DIR_DEFAULT
    return cast(DictConfig, cfg)


def _init_translator() -> None:
    """Instancie l’orchestrateur de message et expose st.session_state['tr'] (domaine streamlit_app)."""
    ss: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)
    lang = str(ss.get(SS_LANG, "fr"))

    # Initialiser l’orchestrateur i18n (API au singulier)
    svc = MessageOrchestrator(
        localedir="i18n/locales",
        domain=I18N_DOMAIN,  # "streamlit_app"
        default_lang=lang,
    )
    svc.load(lang)

    def tr(key: str, **p: Any) -> str:
        # get(key, params: dict[str, Any] | None = None) -> str
        return svc.get(key, p if p else None)

    ss[SS_TR] = cast(Translator, tr)


def _registry() -> Mapping[str, Callable[[], None]]:
    """Retourne le registre des pages multipages."""
    return {
        "NAV_HOME": home.run,
        "NAV_EDA": eda.run,
        "NAV_PIPELINE": pipeline.run,
        "NAV_REPORT": report.run,
        "NAV_NOTEBOOK": notebook.run,
        "NAV_DEMO": demo.run,
    }


def main() -> None:
    """Lance l’application: sidebar config + navigation multipages."""
    _init_defaults()
    if SS_TR not in st.session_state:
        _init_translator()
    tr: Translator = cast(Translator, st.session_state[SS_TR])

    with st.sidebar:
        st.header(tr("SIDEBAR_CONFIG"))
        st.session_state[SS_OUTPUTS_DIR] = st.text_input(tr("LBL_OUTPUTS_DIR"), value=cast(str, st.session_state[SS_OUTPUTS_DIR]))
        st.session_state[SS_PROJECT_NAME] = st.text_input(tr("LBL_PROJECT_NAME"), value=cast(str, st.session_state[SS_PROJECT_NAME]))
        st.session_state[SS_NOTEBOOK_DIR] = st.text_input(tr("LBL_NOTEBOOK_DIR"), value=cast(str, st.session_state[SS_NOTEBOOK_DIR]))
        st.session_state[SS_NOTEBOOK_URL] = st.text_input(tr("LBL_NOTEBOOK_URL"), value=cast(str, st.session_state[SS_NOTEBOOK_URL]))
        new_lang = st.selectbox(tr("LBL_LANGUAGE"), options=SUPPORTED_LANGS, index=SUPPORTED_LANGS.index(cast(str, st.session_state[SS_LANG])))
        if new_lang != st.session_state[SS_LANG]:
            st.session_state[SS_LANG] = new_lang
            _init_translator()
            tr = cast(Translator, st.session_state[SS_TR])

        st.divider()
        st.header(tr("SIDEBAR_NAV"))
        reg = _registry()
        options = [tr(k) for k in NAV_KEYS]
        current = cast(int, st.session_state.get(SS_NAV_INDEX, 0))
        page_label = st.selectbox("Page", options=options, index=min(current, len(options) - 1))
        st.session_state[SS_NAV_INDEX] = options.index(page_label)

        st.divider()
        if st.button(tr("BTN_CLEAR_CACHE")):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success(tr("MSG_CACHES_CLEARED"))

    selected_key = NAV_KEYS[cast(int, st.session_state[SS_NAV_INDEX])]
    _registry()[selected_key]()

    st.caption(
        tr(
            "FOOTER_CONTEXT",
            project=cast(str, st.session_state[SS_PROJECT_NAME]),
            outputs=cast(str, st.session_state[SS_OUTPUTS_DIR]),
            notebook=cast(str, st.session_state[SS_NOTEBOOK_DIR]),
        )
    )


if __name__ == "__main__":
    main()
