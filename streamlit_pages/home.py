from __future__ import annotations

import streamlit as st


def run() -> None:
    st.header("Accueil")
    st.markdown(
        "- Utiliser la barre latérale pour configurer le projet (outputs/projet).  \n"
        "- Naviguer avec les onglets en haut pour EDA, Pipelines et Rapports."
    )
