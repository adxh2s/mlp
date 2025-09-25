from __future__ import annotations

"""Page d'accueil de l'application Streamlit."""

import streamlit as st


def run() -> None:
    """Affiche la page d'accueil et des indications d'usage."""
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.title(tr("TITLE_HOME"))
    st.markdown("- " + tr("SIDEBAR_CONFIG") + " · " + tr("SIDEBAR_NAV"))
