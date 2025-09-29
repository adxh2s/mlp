from __future__ import annotations

"""
Page Démo Streamlit.

Rôle:
- Démonstration interactive (ex. graphique) sans dépendance forte au pipeline.
- Exploite outputs_root et project.name du contexte pour charger/sauvegarder des artefacts.
"""

import random
from pathlib import Path
from typing import Callable, cast

import plotly.graph_objects as go
import streamlit as st

from src.instrumentation.decorators import log_page

SS_CTX = "ctx"
SS_APP_CONFIG = "app_config"


def _models_dir(outputs_dir: str, project_name: str) -> Path:
    """Répertoire de modèles pour la démo."""
    return Path(outputs_dir) / project_name / "models"


def _half_court() -> go.Figure:
    """Crée un demi-terrain (exemple visuel)."""
    fig = go.Figure()
    fig.update_layout(width=800, height=500, xaxis=dict(range=[0, 15]), yaxis=dict(range=[0, 14]), showlegend=True)
    fig.add_shape(type="rect", x0=0, y0=0, x1=15, y1=14, line=dict(color="black"))
    fig.add_shape(type="circle", x0=5, y0=5, x1=10, y1=10, line=dict(color="gray"))
    return fig


@log_page("demo")
def run() -> None:
    """
    Point d'entrée de la page Démo.

    - Affiche une visualisation et des paramètres jouets.
    - Utilise le contexte partagé pour les chemins si besoin.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_DEMO"))

    outputs_dir = cast(str, st.session_state.get(SS_CTX, {}).get("outputs_root", "outputs"))
    project_name = cast(str, getattr(st.session_state.get(SS_APP_CONFIG, None), "project", None).name if st.session_state.get(SS_APP_CONFIG) else "demo_project")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.subheader("Paramètres")
        joueur = st.selectbox(tr("LBL_PLAYER") if callable(tr) else "Joueur", ["A", "B", "C"], index=0)
        essais = st.slider(tr("LBL_ATTEMPTS") if callable(tr) else "Essais", 10, 200, 50, 10)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

    with c2:
        st.subheader("Simulation")
        fig = _half_court()
        random.seed(seed)
        xs = [random.uniform(1, 14) for _ in range(essais)]
        ys = [random.uniform(1, 13) for _ in range(essais)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=f"Tirs {joueur}"))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Models dir: {_models_dir(outputs_dir, project_name)}")
