from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, MutableMapping, cast

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from omegaconf import DictConfig, OmegaConf

"""Page Report Streamlit :
Génère et prévisualise les artefacts produits par le pipeline (HTML, CSV).
Affiche chaque rapport dans des blocs repliables ou onglets.
Diagnostic UX robuste, PEP8 strict et typage fort.
"""

# Constantes (PEP8 - début de script)
SS_CTX = "context"
SS_APP_CONFIG = "app_config"
SS_LOGGER_MANAGER = "logger_manager"
SS_EDA_RESULT = "eda_result"
SS_PIPELINE_RESULT = "pipeline_result"
SS_REPORT_RESULT = "report_result"
SS_MESSAGE_ORCHESTRATOR = "message_orchestrator"

ARTIFACT_PREVIEW_LIMIT = 100  # Nbre de lignes preview CSV
HTML_PREVIEW_HEIGHT = 800     # Hauteur preview du rapport HTML

def _cfg_to_plain_dict(cfg_section: Any) -> dict[str, Any]:
    """Transforme une section OmegaConf/DictConfig/dict en dict natif."""
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

def _render_artifacts(artifacts: list[str]) -> None:
    """
    Affiche tous les artefacts produits : HTML dans des onglets, CSV dans blocs repliables.

    Args:
        artifacts: Liste des chemins vers les artefacts.
    """
    html_files = [p for p in artifacts if isinstance(p, str) and p.lower().endswith(".html")]
    csv_files = [p for p in artifacts if isinstance(p, str) and p.lower().endswith(".csv")]

    if html_files:
        tabs = st.tabs([os.path.basename(p) for p in html_files])
        for tab, path_str in zip(tabs, html_files):
            with tab:
                path = Path(path_str)
                if not path.exists():
                    st.error(f"Fichier HTML introuvable : {path}")
                else:
                    try:
                        components.html(path.read_text(encoding="utf-8"), height=HTML_PREVIEW_HEIGHT, scrolling=True)
                    except Exception as exc:
                        st.error(f"Impossible d'afficher {path} : {exc}")
    else:
        st.info("Aucun rapport HTML produit.")

    for path_str in csv_files:
        with st.expander(f"Pipeline CSV : {os.path.basename(path_str)}", expanded=False):
            path = Path(path_str)
            if not path.exists():
                st.error(f"Fichier CSV introuvable : {path}")
                continue
            try:
                df = pd.read_csv(path)
                st.dataframe(df.head(ARTIFACT_PREVIEW_LIMIT), use_container_width=True)
                st.download_button(
                    label="Télécharger CSV",
                    data=df.to_csv(index=False),
                    file_name=os.path.basename(path_str),
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Impossible de lire {path} : {exc}")

def _run_report(
    report_config: Any,  # typé ReportConfig si importé, Any sinon
    project_dir: str,
    app_config: DictConfig | dict[str, Any],
    context: dict[str, Any],
    logger_manager: Any,
    eda_payload: dict[str, Any],
    pipeline_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Exécute le report orchestrator et retourne le résultat dict à afficher.

    Args:
        report_config: Config du rapport (objet dataclass)
        project_dir: Répertoire projet pour outputs
        app_config: Configuration complète appli
        context: Contexte d'exécution
        logger_manager: gestionnaire log
        eda_payload: résultats EDA
        pipeline_payload: résultats pipeline

    Returns:
        dict des artefacts produits ('main', 'artifacts', ...)
    """
    try:
        from src.orchestrators.report import ReportOrchestrator
    except ImportError:
        raise RuntimeError("Import ReportOrchestrator échoué.")
    rep = ReportOrchestrator(
        cfg=report_config,
        project_dir=project_dir,
        app_config=app_config,
        logger_manager=logger_manager,
        context=context,
    )
    mo = st.session_state.get(SS_MESSAGE_ORCHESTRATOR)
    if mo is not None:
        try:
            rep.attach_message(mo)
        except Exception:
            pass
    return rep.run(eda_payload, pipeline_payload)


def run() -> None:
    """
    Point d'entrée principal de la page Streamlit 'Report' :
    - Génère le rapport complet et affiche tous les artefacts.
    - UX et messages robustes en cas d'échec.
    """
    tr: Callable[[str], str] = cast(Callable[[str], str], st.session_state.get("tr", lambda k, **p: k))
    st.title(tr("TITLE_REPORT"))

    # Récupération de la config applicative
    app_config = cast(DictConfig | dict[str, Any], st.session_state.get(SS_APP_CONFIG, {}))
    if isinstance(app_config, DictConfig):
        rep_section = app_config.get("orchestrators", {}).get("report", {})
    elif isinstance(app_config, dict):
        rep_section = (app_config.get("orchestrators", {}) or {}).get("report", {})
    else:
        rep_section = {}
    try:
        from src.config.schemas import ReportConfig
    except Exception:  # noqa: BLE001
        from src.schemas import ReportConfig  # type: ignore[no-redef]
    report_config_obj = ReportConfig(**_cfg_to_plain_dict(rep_section))

    # Récupération du contexte projet et payloads
    context = cast(dict[str, Any], st.session_state.get(SS_CTX, {}))
    project_dir = str(
        context.get("project_dir")
        or (Path(context.get("outputs_root", "outputs")) / context.get("project_name", "mlp"))
    )
    logger_manager = st.session_state.get(SS_LOGGER_MANAGER)
    eda_payload = cast(MutableMapping[str, Any], st.session_state.get(SS_EDA_RESULT, {}))
    pipeline_payload = cast(MutableMapping[str, Any], st.session_state.get(SS_PIPELINE_RESULT, {"results": []}))

    # Génération du rapport à la demande
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
        except Exception as exc:
            st.error(f"{tr('REPORT_ORCHESTRATOR_FAILED') if callable(tr) else 'Rapport échoué'} — {exc}")

    # Affichage des artefacts produits
    report_result = cast(MutableMapping[str, Any], st.session_state.get(SS_REPORT_RESULT, {}))
    if not report_result:
        st.info("Aucun rapport généré ou artefact vide pour ce projet.")
        return

    artifacts = report_result.get("artifacts") or []
    if artifacts:
        _render_artifacts(artifacts)
    else:
        # Affichage principal/existant fallback
        main = report_result.get("main") or report_result.get("path")
        if isinstance(main, str):
            path = Path(main)
            if path.exists():
                if path.suffix.lower() == ".html":
                    components.html(path.read_text(encoding="utf-8"), height=HTML_PREVIEW_HEIGHT, scrolling=True)
                else:
                    st.markdown(path.read_text(encoding="utf-8"))
