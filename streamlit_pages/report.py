from __future__ import annotations

from typing import Any, Callable, MutableMapping, cast
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from omegaconf import DictConfig, OmegaConf

try:
    from src.config.schemas import ReportConfig
except Exception:  # noqa: BLE001
    from src.schemas import ReportConfig  # type: ignore[no-redef]

from src.orchestrators.report import ReportOrchestrator
from src.instrumentation.decorators import log_page, log_call_ex

# Session state keys
SS_CTX = "context"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_EDA_RESULT = "eda_result"
SS_PIPELINE_RESULT = "pipeline_result"
SS_REPORT_RESULT = "report_result"
SS_MESSAGE_ORCHESTRATOR = "message_orchestrator"


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


@log_call_ex(name="ReportOrchestrator.run")
def _run_report(
    report_config: ReportConfig,
    project_dir: str,
    app_config: DictConfig | dict[str, Any],
    context: dict[str, Any],
    logger_manager: Any,
    eda_payload: dict[str, Any],
    pipeline_payload: dict[str, Any],
):
    rep = ReportOrchestrator(
        cfg=report_config,
        project_dir=project_dir,
        app_config=app_config,              # signature de ReportOrchestrator
        logger_manager=logger_manager,
        context=context,
    )
    if mo := st.session_state.get(SS_MESSAGE_ORCHESTRATOR):
        try:
            rep.attach_message(mo)
        except Exception:
            pass
    return rep.run(eda_payload, pipeline_payload)


@log_page("report")
def run() -> None:
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_REPORT"))

    # Config applicative et section report
    app_config = cast(DictConfig | dict[str, Any], st.session_state.get(SS_APP_CONFIG, {}))
    if isinstance(app_config, DictConfig):
        rep_section = app_config.get("orchestrators", {}).get("report", {})
    elif isinstance(app_config, dict):
        rep_section = (app_config.get("orchestrators", {}) or {}).get("report", {})
    else:
        rep_section = {}
    report_config_obj = ReportConfig(**_cfg_to_plain_dict(rep_section))

    # Contexte, journalisation et payloads
    context = cast(dict[str, Any], st.session_state.get(SS_CTX, {}))
    project_dir = str(
        context.get("project_dir")
        or (Path(context.get("outputs_root", "outputs")) / context.get("project_name", "mlp"))
    )
    logger_manager = st.session_state.get(SS_LOGGER_MANAGER)
    eda_payload = cast(MutableMapping[str, Any], st.session_state.get(SS_EDA_RESULT, {}))
    pipeline_payload = cast(MutableMapping[str, Any], st.session_state.get(SS_PIPELINE_RESULT, {"results": []}))

    # Action utilisateur
    if st.button(tr("BTN_RUN_REPORT") if callable(tr) else "Générer rapport"):
        try:
            result = _run_report(
                report_config_obj,
                project_dir,
                app_config,
                context,
                logger_manager,
                dict(eda_payload),
                dict(pipeline_payload),
            )
            st.session_state[SS_REPORT_RESULT] = result
            st.success(tr("REPORT_DONE") if callable(tr) else "Rapport généré.")
        except Exception as e:
            st.error(f"{tr('REPORT_ORCHESTRATOR_FAILED') if callable(tr) else 'Rapport échoué'} — {e}")

    # Affichage de l'artefact principal
    report_result = cast(MutableMapping[str, Any], st.session_state.get(SS_REPORT_RESULT, {}))
    main = report_result.get("main") or report_result.get("path")
    if isinstance(main, str):
        path = Path(main)
        if path.exists():
            if path.suffix.lower() == ".html":
                components.html(path.read_text(encoding="utf-8"), height=800, scrolling=True)
            else:
                st.markdown(path.read_text(encoding="utf-8"))
