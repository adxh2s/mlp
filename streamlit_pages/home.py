# streamlit_pages/home.py
from __future__ import annotations

"""Page d'accueil de l'application Streamlit."""

import streamlit as st


def run() -> None:
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.set_page_config(page_title=tr("TITLE_HOME"))
    st.header(tr("TITLE_HOME"))
    st.markdown(
        "- " + tr("SIDEBAR_CONFIG") + " · " + tr("SIDEBAR_NAV")
    )
