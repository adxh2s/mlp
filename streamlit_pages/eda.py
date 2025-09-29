from __future__ import annotations

"""
Page EDA Streamlit.

Rôle:
- Consommer le dataset préchargé (X, y) depuis la session.
- Lancer l'EDA via EDAOrchestrator à la demande et mettre en cache le résultat.
"""

from typing import Any, Callable, MutableMapping, cast
from pathlib import Path

import streamlit as st
from src.orchestrators.eda import EDAOrchestrator
from src.instrumentation.decorators import log_page

SS_CTX = "ctx"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_DATA_RESULT = "data_result"
SS_EDA_RESULT = "eda_result"


@log_page("eda")
def run() -> None:
    """
    Point d'entrée de la page EDA.

    - Affiche un aperçu du dataset si disponible.
    - Permet de lancer l'EDA et de consulter les artefacts générés.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_EDA"))

    data = cast(MutableMapping[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    if not data or data.get("X") is None:
        st.info("Aucun dataset préchargé: activer File/Data ou utiliser la page Home pour charger un fichier.")
        return

    X = data.get("X")
    y = data.get("y")
    st.subheader("Aperçu du dataset")
    st.dataframe(X.head(10))

    app_cfg = st.session_state.get(SS_APP_CONFIG)
    if not app_cfg or not app_cfg.orchestrators.eda.enabled:
        st.caption("EDAOrchestrator désactivé dans la configuration.")
        return

    eda_run = st.button(tr("BTN_RUN_EDA") if callable(tr) else "Lancer EDA")
    if eda_run:
        try:
            project_dir = cast(str, st.session_state.get(SS_CTX, {}).get("project_dir", "."))
            eda_orch = EDAOrchestrator(app_cfg.orchestrators.eda, project_dir, logger_manager=st.session_state.get(SS_LOGGER_MANAGER))
            if mo := st.session_state.get("message_orchestrator"):
                eda_orch.attach_message(mo)
            eda_res = eda_orch.run(X, y)
            st.session_state[SS_EDA_RESULT] = eda_res
            st.success("EDA terminée.")
        except Exception as e:  # noqa: BLE001
            st.error(f"EDA échouée: {e}")

    if st.session_state.get(SS_EDA_RESULT):
        with st.expander("Résultats EDA", expanded=False):
            st.json(st.session_state[SS_EDA_RESULT])
