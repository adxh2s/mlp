from __future__ import annotations

import streamlit as st
# Configuration de la page (unique)
APP_TITLE: str = "MLP App"
APP_ICON: str = "📊"
PAGE_LAYOUT: str = "wide"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)

import os
import re
import glob
from pathlib import Path
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

# App-level orchestrators and managers
from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.app import AppOrchestrator  # App bootstrap (Config/Logger/Message/ctx)
from src.orchestrators.file import FileOrchestrator
from src.orchestrators.data import DataOrchestrator
from src.orchestrators.eda import EDAOrchestrator
from src.orchestrators.pipeline import PipelineOrchestrator
from src.orchestrators.report import ReportOrchestrator

# Streamlit pages (must expose a `run()` function)
from streamlit_pages import home, eda, pipeline, report, notebook, demo

"""
Streamlit front-end orchestrator.

Overview
--------
- Bootstraps the application via AppOrchestrator (ConfigManager, LoggerManager,
  MessageOrchestratorApp, and Hydra-safe context).
- On startup (Home page), preloads a dataset by running FileOrchestrator →
  DataOrchestrator if enabled in configuration.
- For each page (Home, EDA, Pipeline, Report, Notebook, Demo), triggers the
  corresponding orchestrator only when needed and caches results in session_state.
- Preserves Markdown docs rendering with image URL rewrites and CSS to constrain width.

Notes
-----
- set_page_config must be called exactly once.
- The code relies on the shared ctx created by ConfigOrchestrator and exposed via AppOrchestrator.
- Orchestrator "enabled" flags in config govern whether each step runs.

"""

# =========================
# UI constants (page setup)
# =========================
APP_TITLE: str = "MLP App"
APP_ICON: str = "📊"
PAGE_LAYOUT: str = "wide"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)  # one call only


# ================
# Session key names
# ================
SS_APP_BOOTSTRAPPED = "app_bootstrapped"
SS_CTX = "ctx"
SS_LOGGER_MANAGER = "logger_manager"
SS_MESSAGE_ORCH = "message_orchestrator"
SS_CONFIG_MANAGER = "config_manager"
SS_APP_CONFIG = "app_config"

SS_FILE_RESULT = "file_result"
SS_DATA_RESULT = "data_result"
SS_EDA_RESULT = "eda_result"
SS_PIPELINE_RESULT = "pipeline_result"
SS_REPORT_RESULT = "report_result"

SS_LANG = "lang"
SS_DOCS_DIR = "docs_dir"
SS_RENDER_DOCS = "render_docs"

# Optional help text
HELP_TEXT = """
Entrée principale Streamlit: bootstrap AppOrchestrator, préchargement File→Data, orchestration par page.
- Contexte partagé: ctx, logger, message, config (en session_state).
- Respect des flags 'enabled' des orchestrateurs.
- Rendu docs Markdown avec réécriture d’images et CSS.
"""


# ===================
# Environment var keys
# ===================
ENV_LANG: str = "MLP_LANG"
ENV_DOCS_DIR: str = "MLP_DOCS_DIR"

# i18n defaults for docs
DEFAULT_I18N_DOMAIN: str = "streamlit_app"
DEFAULT_LOCALES_DIR: str = "i18n/locales"


def _init_ui_defaults() -> None:
    """Initialize minimal UI defaults (language, docs dir) in session_state."""
    ss = cast(MutableMapping[str, Any], st.session_state)
    if SS_LANG not in ss:
        ss[SS_LANG] = os.getenv(ENV_LANG, "fr")
    if SS_DOCS_DIR not in ss:
        ss[SS_DOCS_DIR] = os.getenv(ENV_DOCS_DIR, "docs")


# ===========================
# Markdown image CSS + rewrite
# ===========================
def _inject_md_image_css() -> None:
    """Constrain all Markdown images to column width."""
    st.markdown(
        "<style>.stMarkdown img{max-width:100%;height:auto;}</style>",
        unsafe_allow_html=True,
    )


IMG_MD_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)')
IMG_HTML_RE = re.compile(r'<img\s+[^>]*src=["\'](?P<src>[^"\']+)["\'][^>]*>')


def _rewrite_image_urls(md_text: str, md_file: str, docs_base: str = "docs") -> str:
    """
    Rewrite relative image URLs to app/static/docs/... for Streamlit static serving.

    Examples
    --------
    images/xxx.png → app/static/docs/{lang}/{section}/images/xxx.png
    ./images/xxx.png → app/static/docs/{lang}/{section}/images/xxx.png

    """
    md_dir = Path(md_file).parent
    md_dir_rel = md_dir.relative_to(docs_base)  # ex: fr/home
    docs_url_root = os.path.basename(os.path.normpath(docs_base)) if os.path.isabs(docs_base) else docs_base
    static_base = f"app/static/{docs_url_root}/{md_dir_rel.as_posix()}/"

    def _rewrite_src(src: str) -> str:
        s = src.strip().strip('\'"')
        if s.startswith(("http://", "https://", "app/static/")):
            return src
        if s.startswith(("./images/", "images/")):
            s2 = s.lstrip("./")
            return f"{static_base}{s2}"
        return src

    def md_sub(m: re.Match) -> str:
        alt = m.group("alt")
        src = m.group("src")
        if " " in src and not src.strip().startswith(("http://", "https://", "app/static/")):
            p, _, title = src.strip().partition(" ")
            new = _rewrite_src(p)
            return f'![{alt}]({new} {title})'
        new = _rewrite_src(src)
        return f'![{alt}]({new})'

    def html_sub(m: re.Match) -> str:
        src = m.group("src")
        new = _rewrite_src(src)
        return m.group(0).replace(src, new)

    md_text = IMG_MD_RE.sub(md_sub, md_text)
    md_text = IMG_HTML_RE.sub(html_sub, md_text)
    return md_text


def render_docs(section: str, lang: str | None = None) -> None:
    """
    Render Markdown documentation for a given section and current language.

    Search order
    ------------
    docs/{lang}/{section}/*.md
    docs/{section}.{lang}.*.md
    docs/{section}.*.{lang}.md
    docs/{section}.*.md
    docs/{section}/*.md
    """
    base = cast(str, st.session_state.get(SS_DOCS_DIR, "docs"))
    cur_lang = (lang or cast(str, st.session_state.get(SS_LANG, os.getenv(ENV_LANG, "fr")))).lower()
    patterns = [
        os.path.join(base, cur_lang, section, "*.md"),
        os.path.join(base, f"{section}.{cur_lang}.*.md"),
        os.path.join(base, f"{section}.*.{cur_lang}.md"),
        os.path.join(base, f"{section}.*.md"),
        os.path.join(base, section, "*.md"),
    ]
    seen: set[str] = set()
    candidates: list[str] = []
    for pat in patterns:
        for path in glob.glob(pat):
            if path not in seen:
                seen.add(path)
                candidates.append(path)

    _inject_md_image_css()
    for path in sorted(candidates, key=lambda p: os.path.basename(p)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            content = _rewrite_image_urls(content, md_file=path, docs_base=base)
            st.markdown(content, unsafe_allow_html=False)
        except Exception as e:
            st.warning(f"Impossible de lire {path}: {e}")


# =======================
# App bootstrap and cache
# =======================
def _bootstrap_app() -> None:
    """
    Initialize AppOrchestrator and cache context and managers in session_state.

    Steps
    -----
    1) Load Hydra DictConfig via ConfigManager.
    2) Build AppOrchestrator (Logger, Config, Message, ctx).
    3) Store ctx, logger, message and app config in session_state.
    """
    if st.session_state.get(SS_APP_BOOTSTRAPPED, False):
        return

    # Load base config (Hydra DictConfig)
    hydra_cfg: DictConfig = ConfigManager.load_base_config()
    app = AppOrchestrator(hydra_cfg)

    # Cache shared objects
    st.session_state[SS_CONFIG_MANAGER] = app.config_manager
    st.session_state[SS_LOGGER_MANAGER] = app.logger_manager
    st.session_state[SS_MESSAGE_ORCH] = app.message_orchestrator
    st.session_state[SS_CTX] = app.ctx

    # Application config model (AppConfig)
    app_cfg = app.config_orchestrator.get_app_config()
    st.session_state[SS_APP_CONFIG] = app_cfg

    # Expose docs renderer for pages
    st.session_state[SS_RENDER_DOCS] = render_docs

    # Minimal UI defaults
    _init_ui_defaults()

    st.session_state[SS_APP_BOOTSTRAPPED] = True


def _preload_file_and_data() -> None:
    """
    On app load (Home), attempt to preload a dataset via FileOrchestrator → DataOrchestrator.

    - Respects orchestrator flags 'enabled' in the app configuration.
    - Stores results under SS_FILE_RESULT and SS_DATA_RESULT in session_state.
    - Non-blocking: logs and continues on errors.
    """
    if st.session_state.get(SS_DATA_RESULT):
        return  # already preloaded

    app_cfg = cast(Any, st.session_state.get(SS_APP_CONFIG))
    logger_manager = cast(Any, st.session_state.get(SS_LOGGER_MANAGER))
    ctx = cast(dict[str, str], st.session_state.get(SS_CTX, {}))

    try:
        # File
        if app_cfg and app_cfg.orchestrators.file and app_cfg.orchestrators.file.enabled:
            file_orch = FileOrchestrator(app_cfg.orchestrators.file, logger_manager=logger_manager, ctx=ctx)
            file_result = file_orch.process_input()
            st.session_state[SS_FILE_RESULT] = file_result
        else:
            # No file orchestrator (disabled); nothing to preload
            return

        # Data
        if app_cfg.orchestrators.data and app_cfg.orchestrators.data.enabled:
            if file_result := st.session_state.get(SS_FILE_RESULT):
                if file_result.get("found") and file_result.get("data") is not None:
                    data_orch = DataOrchestrator(app_cfg.orchestrators.data, logger_manager=logger_manager)
                    # Attach message bus if present
                    if mo := st.session_state.get(SS_MESSAGE_ORCH):
                        data_orch.attach_message(mo)
                    data_result = data_orch.run(file_result["data"])
                    st.session_state[SS_DATA_RESULT] = data_result
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Préchargement File→Data ignoré (raison: {exc}).")


# ===============
# Sidebar and nav
# ===============
def _pages_registry() -> "OrderedDict[str, Callable[[], None]]":
    """
    Register Streamlit pages; each page must define a `run()` function.

    Returns
    -------
    OrderedDict: label → callable
    """
    return OrderedDict(
        [
            ("Home", home.run),
            ("EDA", eda.run),
            ("Pipelines", pipeline.run),
            ("Rapports", report.run),
            ("Notebooks", notebook.run),
            ("Démo", demo.run),
        ]
    )


def _sidebar(pages: "OrderedDict[str, Callable[[], None]]") -> str:
    """
    Render the sidebar with project info, controls, and page selector.

    Returns
    -------
    str: selected page label
    """
    app_cfg = cast(Any, st.session_state.get(SS_APP_CONFIG))
    with st.sidebar:
        st.header(APP_TITLE)
        page_label = st.selectbox(label="Page", options=list(pages.keys()), index=0)

        # Optional quick controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Vider le cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                for k in (SS_FILE_RESULT, SS_DATA_RESULT, SS_EDA_RESULT, SS_PIPELINE_RESULT, SS_REPORT_RESULT):
                    st.session_state.pop(k, None)
                st.success("Cache vidé.")
        with col2:
            if st.button("Aide"):
                st.info(HELP_TEXT)

        # Small status
        if app_cfg:
            st.caption(
                f"Projet: {app_cfg.project.name} | Orchestrateurs: "
                f"file={'on' if app_cfg.orchestrators.file.enabled else 'off'}, "
                f"data={'on' if app_cfg.orchestrators.data.enabled else 'off'}, "
                f"eda={'on' if app_cfg.orchestrators.eda.enabled else 'off'}, "
                f"pipeline={'on' if app_cfg.orchestrators.pipeline.enabled else 'off'}, "
                f"report={'on' if app_cfg.orchestrators.report.enabled else 'off'}"
            )

        return page_label


# =======================
# Per-page orchestration
# =======================
def _ensure_eda() -> None:
    """Run EDAOrchestrator once per session if enabled and if data is available."""
    if st.session_state.get(SS_EDA_RESULT):
        return
    app_cfg = cast(Any, st.session_state.get(SS_APP_CONFIG))
    logger_manager = cast(Any, st.session_state.get(SS_LOGGER_MANAGER))
    data = cast(dict[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    if not app_cfg or not app_cfg.orchestrators.eda.enabled or not data:
        return
    X, y = data.get("X"), data.get("y")
    if X is None:
        return
    try:
        eda_orch = EDAOrchestrator(app_cfg.orchestrators.eda, project_dir=st.session_state[SS_CTX]["project_dir"], logger_manager=logger_manager)
        if mo := st.session_state.get(SS_MESSAGE_ORCH):
            eda_orch.attach_message(mo)
        st.session_state[SS_EDA_RESULT] = eda_orch.run(X, y)
    except Exception as exc:  # noqa: BLE001
        st.error(f"EDA échouée: {exc}")


def _ensure_pipeline() -> None:
    """Run PipelineOrchestrator once per session if enabled and if target is available."""
    if st.session_state.get(SS_PIPELINE_RESULT):
        return
    app_cfg = cast(Any, st.session_state.get(SS_APP_CONFIG))
    logger_manager = cast(Any, st.session_state.get(SS_LOGGER_MANAGER))
    ctx = cast(dict[str, str], st.session_state.get(SS_CTX, {}))
    data = cast(dict[str, Any], st.session_state.get(SS_DATA_RESULT, {}))
    if not app_cfg or not app_cfg.orchestrators.pipeline.enabled or not data:
        return
    X, y = data.get("X"), data.get("y")
    if X is None or y is None:
        return
    try:
        # Resolve pipeline out_dir similar to GeneralOrchestrator logic
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

        pipes = PipelineOrchestrator(
            app_cfg.orchestrators.pipeline,
            project_dir=st.session_state[SS_CTX]["project_dir"],
            random_state=app_cfg.project.random_state,
            logger_manager=logger_manager,
            out_dir=out_dir,
            ctx=ctx,
        )
        if mo := st.session_state.get(SS_MESSAGE_ORCH):
            pipes.attach_message(mo)
        st.session_state[SS_PIPELINE_RESULT] = pipes.run(X, y)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Pipeline échouée: {exc}")


def _ensure_report() -> None:
    """Run ReportOrchestrator once per session if enabled and when EDA or Pipeline results exist."""
    if st.session_state.get(SS_REPORT_RESULT):
        return
    app_cfg = cast(Any, st.session_state.get(SS_APP_CONFIG))
    logger_manager = cast(Any, st.session_state.get(SS_LOGGER_MANAGER))
    ctx = cast(dict[str, str], st.session_state.get(SS_CTX, {}))
    if not app_cfg or not app_cfg.orchestrators.report.enabled:
        return
    try:
        rep = ReportOrchestrator(
            app_cfg.orchestrators.report,
            st.session_state[SS_CTX]["project_dir"],
            app_cfg,
            logger_manager=logger_manager,
            ctx=ctx,
        )
        if mo := st.session_state.get(SS_MESSAGE_ORCH):
            rep.attach_message(mo)
        eda_payload = cast(dict[str, Any], st.session_state.get(SS_EDA_RESULT, {}))
        pipeline_payload = cast(dict[str, Any], st.session_state.get(SS_PIPELINE_RESULT, {"results": []}))
        st.session_state[SS_REPORT_RESULT] = rep.run(eda_payload, pipeline_payload)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Rapport échoué: {exc}")


# =========
# Main flow
# =========
def main() -> None:
    """
    Streamlit entrypoint: bootstrap app, preload data, and route to selected page.

    Flow
    ----
    - Bootstrap shared context once (AppOrchestrator).
    - Preload dataset via File→Data if enabled (Home).
    - Depending on selected page, ensure orchestrations (EDA, Pipeline, Report).
    - Delegate visual rendering to page modules, which consume session_state.
    """
    _bootstrap_app()
    pages = _pages_registry()
    page_label = _sidebar(pages)

    # Page title
    st.title(APP_TITLE)

    # Preload only once at app start (Home first render)
    if page_label == "Home":
        _preload_file_and_data()

    # Trigger per-page orchestrations (compute → then render)
    if page_label == "EDA":
        _ensure_eda()
    elif page_label == "Pipelines":
        _ensure_pipeline()
    elif page_label == "Rapports":
        # Report can rely on EDA and Pipeline if available
        _ensure_eda()
        _ensure_pipeline()
        _ensure_report()

    # Render selected page
    runner = pages.get(page_label)
    if runner is None:
        st.error(f"Page inconnue: {page_label}")
        return
    # Pages read from session_state (ctx, data_result, eda_result, etc.)
    runner()


if __name__ == "__main__":
    main()
