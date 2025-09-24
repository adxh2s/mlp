# streamlit_pages/demo.py
from __future__ import annotations

"""Page de démonstration: simulation de tir en match avec demi-terrain."""

import random
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


def _models_dir(outputs_dir: str, project_name: str) -> Path:
    return Path(outputs_dir) / project_name / "models"


def _half_court() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(width=800, height=500, xaxis=dict(range=[0, 15]), yaxis=dict(range=[0, 14]), showlegend=True)
    fig.add_shape(type="rect", x0=0, y0=0, x1=15, y1=14, line=dict(color="black"))
    fig.add_shape(type="circle", x0=5, y0=5, x1=10, y1=10, line=dict(color="gray"))
    return fig


def run() -> None:
    tr = st.session_state.get("tr", lambda k, **p: k)
    st.set_page_config(page_title=tr("TITLE_DEMO"), layout="wide")
    st.title(tr("TITLE_DEMO"))

    outputs_dir = st.session_state.get("outputs_dir", "outputs")
    project_name = st.session_state.get("project_name", "demo_project")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.subheader("Paramètres")
        joueur = st.selectbox(tr("LBL_SHOOTER"), ["Joueur A", "Joueur B", "Joueur C"])
        nb_def = st.select_slider(tr("LBL_DEFENDERS"), options=[0, 1, 2], value=1)
        temps = st.slider(tr("LBL_TIME"), 0, 24, 12, 1)
        _ = st.selectbox(tr("LBL_DRIBBLE"), ["Non", "Oui"])
        _ = st.selectbox(tr("LBL_SHOT_TYPE"), ["Layup", "Mi-distance", "3 points", "Step-back"])
        _ = st.selectbox(tr("LBL_FATIGUE"), ["0%", "25%", "50%", "75%", "90%"])
        lancer = st.button(tr("BTN_RUN_SIM"))

    with c2:
        st.subheader(tr("LBL_HALFCOURT"))
        fig = _half_court()
        if lancer:
            x, y = [7.5], [1.0]
            steps = max(1, int(temps / 2))
            for _ in range(steps):
                x.append(max(0, min(15, x[-1] + random.uniform(-0.8, 0.8))))
                y.append(max(0, min(14, y[-1] + random.uniform(0.2, 1.2))))
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=f"Tireur: {joueur}"))
            for d in range(int(nb_def)):
                dx, dy = [random.uniform(5, 10)], [random.uniform(6, 12)]
                for _ in range(steps):
                    dx.append(max(0, min(15, dx[-1] + random.uniform(-0.6, 0.6))))
                    dy.append(max(0, min(14, dy[-1] + random.uniform(-0.6, 0.6))))
                fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines+markers", name=f"Défenseur {d+1}"))
            st.success(tr("MSG_PLACEHOLDER_PROBA"))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Models dir: {_models_dir(outputs_dir, project_name)}")
