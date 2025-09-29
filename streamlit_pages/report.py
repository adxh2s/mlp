from __future__ import annotations

"""
Page Report Streamlit.

Rôle:
- Générer un rapport à partir des résultats EDA/Pipeline disponibles.
- Afficher le contenu si le rapport est textuel ou HTML.
"""

from typing import Any, Callable, MutableMapping, cast
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.orchestrators.report import ReportOrchestrator
from src.instrumentation.decorators import log_page

SS_CTX = "ctx"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_EDA_RESULT = "eda_result"
SS_PIPELINE_RESULT = "pipeline_result"
SS_REPORT_RESULT = "report_result"


@log_page("report")
def run() -> None:
    """
    Point d'entrée de la page Rapports.

    - Permet de générer le rapport à partir des résultats EDA/Pipeline.
    - Met en cache le résultat et affiche le rendu si applicable.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_REPORT"))

    app_cfg = st.session_state.get(SS_APP_CONFIG)
    if not app_cfg or not app_cfg.orchestrators.report.enabled:
        st.caption("ReportOrchestrator désactivé dans la configuration.")
        return

    eda_payload = cast(MutableMapping[str, Any], st.session_state.get(SS_EDA_RESULT, {}))
    pipeline_payload = cast(MutableMapping[str, Any], st.session_state.get(SS_PIPELINE_RESULT, {"results": []}))

    if st.button(tr("BTN_RUN_REPORT") if callable(tr) else "Générer rapport"):
        try:
            rep = ReportOrchestrator(
                app_cfg.orchestrators.report,
                st.session_state[SS_CTX]["project_dir"],
                app_cfg,
                logger_manager=st.session_state.get(SS_LOGGER_MANAGER),
                ctx=cast(dict[str, str], st.session_state.get(SS_CTX, {})),
            )
            if mo := st.session_state.get("message_orchestrator"):
                rep.attach_message(mo)
            result = rep.run(eda_payload, pipeline_payload)
            st.session_state[SS_REPORT_RESULT] = result
            st.success(tr("REPORT_DONE") if callable(tr) else "Rapport généré.")
        except Exception as e:  # noqa: BLE001
            st.error(f"{tr('REPORT_ORCHESTRATOR_FAILED') if callable(tr) else 'Rapport échoué'}: {e}")

    # Affichage si un chemin principal est renvoyé (optionnel selon implémentation)
    res = cast(MutableMapping[str, Any], st.session_state.get(SS_REPORT_RESULT, {}))
    artifacts = res.get("artifacts")
    main = res.get("main") or res.get("path")
    if isinstance(main, str):
        p = Path(main)
        if p.exists():
            if p.suffix.lower() == ".html":
                components.html(p.read_text(encoding="utf-8"), height=800, scrolling=True)
            else:
                st.markdown(p.read_text(encoding="utf-8"))
