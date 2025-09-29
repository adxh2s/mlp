from __future__ import annotations

"""
Page Pipeline Streamlit.

Rôle:
- Déclencher le pipeline ML (cross-val, entraînement) sur X,y préchargés.
- Respecter la résolution de out_dir identique au GeneralOrchestrator.
"""

from typing import Any, Callable, MutableMapping, cast
from pathlib import Path

import streamlit as st
from src.orchestrators.pipeline import PipelineOrchestrator
from src.instrumentation.decorators import log_page

SS_CTX = "ctx"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_DATA_RESULT = "data_result"
SS_PIPELINE_RESULT = "pipeline_result"


@log_page("pipeline")
def run() -> None:
    """
    Point d'entrée de la page Pipeline.

    - Vérifie la présence de X,y (cible requise).
    - Lance le pipeline et affiche un résumé JSON des résultats.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_PIPELINES"))

    data = cast(MutableMapping[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    X = data.get("X")
    y = data.get("y")
    if X is None or y is None:
        st.info(tr("MSG_NO_INPUT_FILE") if callable(tr) else "Aucun dataset avec cible disponible.")
        return

    app_cfg = st.session_state.get(SS_APP_CONFIG)
    if not app_cfg or not app_cfg.orchestrators.pipeline.enabled:
        st.caption("PipelineOrchestrator désactivé dans la configuration.")
        return

    # Résoudre out_dir comme dans GeneralOrchestrator
    p_out_cfg = getattr(app_cfg.orchestrators.pipeline, "out_dir", None)
    if p_out_cfg:
        p = Path(p_out_cfg)
        if p.is_absolute():
            out_dir = str(p)
        elif p.parts and p.parts[0] == "outputs":
            root_dir = Path(st.session_state[SS_CTX]["project_dir"]).parent.parent
            out_dir = str(root_dir / p_out_cfg)
        else:
            out_dir = str(Path(st.session_state[SS_CTX]["project_dir"]) / p_out_cfg)
    else:
        out_dir = str(Path(st.session_state[SS_CTX]["project_dir"]) / "pipeline_cv")

    if st.button(tr("BTN_RUN_PIPELINE") if callable(tr) else "Lancer pipeline"):
        try:
            po = PipelineOrchestrator(
                app_cfg.orchestrators.pipeline,
                project_dir=st.session_state[SS_CTX]["project_dir"],
                random_state=app_cfg.project.random_state,
                logger_manager=st.session_state.get(SS_LOGGER_MANAGER),
                out_dir=out_dir,
                ctx=cast(dict[str, str], st.session_state.get(SS_CTX, {})),
            )
            if mo := st.session_state.get("message_orchestrator"):
                po.attach_message(mo)
            res = po.run(X, y)
            st.session_state[SS_PIPELINE_RESULT] = res
            st.success(tr("PIPELINE_DONE") if callable(tr) else "Pipeline terminé.")
            st.json(res)
        except Exception as e:  # noqa: BLE001
            st.error(f"{tr('PIPELINE_ORCHESTRATOR_FAILED') if callable(tr) else 'Pipeline échoué'}: {e}")
