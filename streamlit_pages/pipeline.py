from __future__ import annotations

from typing import Any, Callable, MutableMapping, cast
from pathlib import Path

import streamlit as st
from omegaconf import DictConfig, OmegaConf

try:
    from src.config.schemas import PipelineConfig
except Exception:  # noqa: BLE001
    from src.schemas import PipelineConfig  # type: ignore[no-redef]

from src.orchestrators.pipeline import PipelineOrchestrator
from src.instrumentation.decorators import log_page, log_call_ex, summarize_df_y

SS_CTX = "context"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_DATA_RESULT = "data_result"
SS_PIPELINE_RESULT = "pipeline_result"

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

@log_call_ex(name="PipelineOrchestrator.run", arg_summary=summarize_df_y)
def _run_pipeline(cfg: PipelineConfig, project_dir: str, context: dict[str, Any], logger_manager: Any, X, y):
    po = PipelineOrchestrator(
        cfg=cfg,
        project_dir=project_dir,
        random_state=int(context.get("random_state") or 42),
        logger_manager=logger_manager,
    )
    if mo := st.session_state.get("message_orchestrator"):
        po.attach_message(mo)
    return po.run(X, y)

@log_page("pipeline")
def run() -> None:
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_PIPELINES", None))

    data = cast(MutableMapping[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    X = data.get("X")
    y = data.get("y")
    if X is None or y is None:
        st.info(tr("MSG_NO_INPUT_FILE", None) if callable(tr) else "Aucun dataset avec cible disponible.")
        return

    app_config = cast(DictConfig | dict[str, Any], st.session_state.get(SS_APP_CONFIG, {}))
    if isinstance(app_config, DictConfig):
        pl_section = app_config.get("orchestrators", {}).get("pipeline", {})
    elif isinstance(app_config, dict):
        pl_section = (app_config.get("orchestrators", {}) or {}).get("pipeline", {})
    else:
        pl_section = {}
    pl_plain = _cfg_to_plain_dict(pl_section)

    context = cast(dict[str, Any], st.session_state.get(SS_CTX, {}))
    project_dir = Path(context.get("project_dir") or (Path(context.get("outputs_root", "outputs")) / context.get("project_name", "mlp")))

    p_out_cfg = pl_plain.get("out_dir")
    if p_out_cfg:
        p = Path(p_out_cfg)
        if p.is_absolute():
            out_dir = p
        elif p.parts and p.parts[0] == "outputs":
            out_dir = project_dir.parent.parent / p_out_cfg
        else:
            out_dir = project_dir / p_out_cfg
    else:
        out_dir = project_dir / "pipeline"
    pl_plain["out_dir"] = str(out_dir)

    pipeline_cfg_obj = PipelineConfig(**pl_plain)
    if not pipeline_cfg_obj.enabled:
        st.caption("PipelineOrchestrator désactivé dans la configuration.")
        return

    if st.button(tr("BTN_RUN_PIPELINE", None) if callable(tr) else "Lancer pipeline"):
        try:
            res = _run_pipeline(pipeline_cfg_obj, str(project_dir), context, st.session_state.get(SS_LOGGER_MANAGER), X, y)
            st.session_state[SS_PIPELINE_RESULT] = res
            st.success(tr("PIPELINE_DONE", None) if callable(tr) else "Pipeline terminé.")
            st.json(res)
        except Exception as e:
            st.error(f"{tr('PIPELINE_ORCHESTRATOR_FAILED') if callable(tr) else 'Pipeline échoué'} — {e}")
