# streamlit_app.py
from __future__ import annotations

import os
from collections.abc import Callable

import streamlit as st

from src.orchestrators.messages import MessagesOrchestrator
from streamlit_pages import demo, eda, home, notebooks, pipelines, reports

"""Entrée principale Streamlit avec i18n gettext + navigation multipages."""

NAV_KEYS: list[str] = ["NAV_HOME", "NAV_EDA", "NAV_PIPELINES", "NAV_REPORTS", "NAV_NOTEBOOKS", "NAV_DEMO"]
SUPPORTED_LANGS: list[str] = ["fr", "en"]


def _registry(tr: Callable[[str], str]) -> dict[str, Callable[[], None]]:
    """Mappe les clés de navigation vers les pages; labels via tr()."""
    return {
        "NAV_HOME": home.run,
        "NAV_EDA": eda.run,
        "NAV_PIPELINES": pipelines.run,
        "NAV_REPORTS": reports.run,
        "NAV_NOTEBOOKS": notebooks.run,
        "NAV_DEMO": demo.run,
    }


def _init_defaults() -> None:
    defaults = {
        "outputs_dir": os.getenv("MLP_OUTPUTS_DIR", "outputs"),
        "project_name": os.getenv("MLP_PROJECT_NAME", "demo_project"),
        "notebooks_dir": os.getenv("MLP_NOTEBOOKS_DIR", "notebooks"),
        "notebooks_url": os.getenv("MLP_NOTEBOOKS_URL", ""),
        "lang": os.getenv("MLP_LANG", "fr"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _init_messages() -> None:
    """Initialise l'orchestrateur de messages et expose st.session_state['tr']."""
    lang = str(st.session_state.get("lang", "fr"))
    svc = MessagesOrchestrator(localedir="i18n/locales", domain="streamlit_app", default_lang=lang)
    svc.load(lang)
    st.session_state["tr"] = lambda key, **p: svc.get(key, p)


def main() -> None:
    _init_defaults()
    if "tr" not in st.session_state:
        _init_messages()
    tr = st.session_state["tr"]

    st.set_page_config(page_title=tr("APP_TITLE"), page_icon="📊", layout="wide")

    with st.sidebar:
        st.header(tr("SIDEBAR_CONFIG"))
        st.session_state["outputs_dir"] = st.text_input(tr("LBL_OUTPUTS_DIR"), value=st.session_state["outputs_dir"])
        st.session_state["project_name"] = st.text_input(tr("LBL_PROJECT_NAME"), value=st.session_state["project_name"])
        st.session_state["notebooks_dir"] = st.text_input(tr("LBL_NOTEBOOKS_DIR"), value=st.session_state["notebooks_dir"])
        st.session_state["notebooks_url"] = st.text_input(tr("LBL_NOTEBOOKS_URL"), value=st.session_state["notebooks_url"])
        new_lang = st.selectbox(tr("LBL_LANGUAGE"), options=SUPPORTED_LANGS, index=SUPPORTED_LANGS.index(st.session_state["lang"]))
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
            _init_messages()
            tr = st.session_state["tr"]

        st.divider()
        st.header(tr("SIDEBAR_NAV"))
        reg = _registry(tr)
        options = [tr(k) for k in NAV_KEYS]
        current = st.session_state.get("nav_index", 0)
        page_label = st.selectbox("Page", options=options, index=min(current, len(options) - 1))
        st.session_state["nav_index"] = options.index(page_label)

        st.divider()
        if st.button(tr("BTN_CLEAR_CACHE")):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success(tr("MSG_CACHES_CLEARED"))

    selected_key = NAV_KEYS[st.session_state["nav_index"]]
    reg[selected_key]()

    st.caption(
        tr(
            "FOOTER_CONTEXT",
            project=st.session_state["project_name"],
            outputs=st.session_state["outputs_dir"],
            notebooks=st.session_state["notebooks_dir"],
        )
    )


if __name__ == "__main__":
    main()
