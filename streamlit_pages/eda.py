from __future__ import annotations

from typing import Any, Callable, MutableMapping, cast
from pathlib import Path

import pandas as pd
import streamlit as st
from omegaconf import DictConfig, OmegaConf

# Import robuste des schémas Pydantic (selon l’arborescence du projet)
try:
    from src.config.schemas import EDAConfig
except Exception:  # noqa: BLE001
    from src.schemas import EDAConfig  # type: ignore[no-redef]

from src.orchestrators.eda import EDAOrchestrator
from src.instrumentation.decorators import log_page, log_call_ex, summarize_df_y  # <-- décorateurs centraux

SS_CTX = "context"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_DATA_RESULT = "data_result"
SS_EDA_RESULT = "eda_result"

def _cfg_to_plain_dict(cfg_section: Any) -> dict[str, Any]:
    if isinstance(cfg_section, DictConfig):
        return cast(dict[str, Any], OmegaConf.to_container(cfg_section, resolve=True))
    if isinstance(cfg_section, dict):
        return cfg_section
    if cfg_section is None:
        return {}
    try:
        return dict(cfg_section)  # type: ignore[arg-type]
    except Exception:
        return {}

@log_call_ex(name="EDAOrchestrator.run", arg_summary=summarize_df_y)
def _run_eda(eda_cfg: EDAConfig, project_dir: str, context: dict[str, Any], logger_manager: Any, X: pd.DataFrame, y: pd.Series | None):
    eda = EDAOrchestrator(
        cfg=eda_cfg,
        project_dir=project_dir,
        logger_manager=logger_manager,
    )
    if hasattr(eda, "attach_message") and callable(getattr(eda, "attach_message")):
        mo = st.session_state.get("message_orchestrator")
        if mo:
            eda.attach_message(mo)
    return eda.run(X, y)

@log_page("eda")
def run() -> None:
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_EDA"))

    data = cast(MutableMapping[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    if not data or data.get("X") is None:
        st.info(tr("MSG_NO_DATA_EDA") if callable(tr) else "Aucun dataset chargé.")
        return

    X = cast(pd.DataFrame, data.get("X"))
    y = cast(pd.Series | None, data.get("y"))

    with st.expander(tr("LBL_DATA_PREVIEW") if callable(tr) else "Aperçu du dataset", expanded=True):
        n_show = st.number_input("Lignes à afficher", min_value=5, max_value=100, value=10, step=5)
        st.write(X.head(int(n_show)))
        st.write({"shape": tuple(getattr(X, "shape", (None, None))), "y_present": y is not None})

    if st.button(tr("BTN_RUN_EDA") if callable(tr) else "Lancer EDA", type="primary"):
        # Préparer la config Pydantic
        app_config = cast(DictConfig | dict[str, Any], st.session_state.get(SS_APP_CONFIG, {}))
        if isinstance(app_config, DictConfig):
            eda_section = app_config.get("orchestrators", {}).get("eda", {})
        elif isinstance(app_config, dict):
            eda_section = (app_config.get("orchestrators", {}) or {}).get("eda", {})
        else:
            eda_section = {}
        eda_cfg_obj = EDAConfig(**_cfg_to_plain_dict(eda_section))

        context = cast(dict[str, Any], st.session_state.get(SS_CTX, {}))
        project_dir = str(context.get("project_dir") or (Path(context.get("outputs_root", "outputs")) / context.get("project_name", "mlp")))
        logger_manager = st.session_state.get(SS_LOGGER_MANAGER)

        try:
            result = _run_eda(eda_cfg_obj, project_dir, context, logger_manager, X, y)
            st.session_state[SS_EDA_RESULT] = result
            st.success(tr("MSG_EDA_DONE") if callable(tr) else "EDA terminée.")
            with st.expander(tr("LBL_EDA_RESULT") if callable(tr) else "Résultat EDA", expanded=False):
                st.json(result if isinstance(result, dict) else {"result": str(result)})
        except Exception as e:  # le décorateur a déjà logué l’erreur avec la pile
            st.error(f"{tr('MSG_EDA_FAILED') if callable(tr) else 'EDA échouée'} — {e}")
