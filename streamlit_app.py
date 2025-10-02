from __future__ import annotations

import streamlit as st

# -------------------- Page config (doit être le premier appel Streamlit) --------------------
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

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

# Orchestrateur d'application (logger + config + message)
from src.orchestrators.app import AppOrchestrator

# Instrumentation utilitaire (préchargement Data + logs décorés)
from src.instrumentation.data_manager import DataManager
from src.instrumentation.decorators import log_call_ex, summarize_df_y

# -------------------- Pages (doivent exposer run()) --------------------
from streamlit_pages import demo, eda, home, notebook, pipeline, report, logs  # noqa: E402

# -------------------- Help UI --------------------
HELP_TEXT = """
Entrée principale Streamlit: i18n via gettext, navigation multipages, config via AppOrchestrator.
- set_page_config appelé une seule fois.
- i18n centralisée (AppOrchestrator → Message) → st.session_state['tr'].
- Compatibilité Docker: conf/config.yaml et i18n/locales montés, avec fallback contrôlé.
"""

# -------------------- Clés session_state & env --------------------
SS_OUTPUTS_DIR: str = "outputs_dir"
SS_PROJECT_NAME: str = "project_name"
SS_NOTEBOOK_DIR: str = "notebooks_dir"
SS_NOTEBOOK_URL: str = "notebooks_url"
SS_LANG: str = "lang"
SS_DOCS_DIR: str = "docs_dir"

# Clés partagées avec les pages et l’app standalone
SS_CTX: str = "context"
SS_APP_CONFIG: str = "app_config"
SS_LOGGER_MANAGER: str = "logger_manager"
SS_DATA_RESULT: str = "data_result"
SS_RENDER_DOCS: str = "render_docs"
SS_LOG_FILE: str = "log_file"

ENV_OUTPUTS_DIR: str = "MLP_OUTPUTS_DIR"
ENV_PROJECT_NAME: str = "MLP_PROJECT_NAME"
ENV_NOTEBOOK_DIR: str = "MLP_NOTEBOOKS_DIR"
ENV_NOTEBOOK_URL: str = "MLP_NOTEBOOKS_URL"
ENV_LANG: str = "MLP_LANG"
ENV_DOCS_DIR: str = "MLP_DOCS_DIR"
ENV_LOG_FILE: str = "MLP_LOG_FILE"

DEFAULT_I18N_DOMAIN: str = "streamlit_app"
DEFAULT_LOCALES_DIR: str = "i18n/locales"

# -------------------- Construction AppOrchestrator (Hydra + fallback) --------------------
def _build_app_orchestrator() -> tuple[AppOrchestrator | None, DictConfig | None, str | None]:
    """
    1) Essaye: initialize_config_dir + compose + HydraConfig.set_config + AppOrchestrator dans le même bloc.
    2) Fallback: OmegaConf.load si config.yaml direct est autosuffisant.
    3) Fallback: config minimale incluant orchestrators.
    """
    last_err: str | None = None

    # 1) Composition Hydra + init orchestrateur dans le même contexte
    for conf_dir in ("conf", "/app/conf"):
        cfg_path = Path(conf_dir) / "config.yaml"
        if cfg_path.is_file():
            try:
                # Clear pour éviter "Hydra is already initialized" à chaque rerun Streamlit
                GlobalHydra.instance().clear()
                with initialize_config_dir(version_base=None, config_dir=str(Path(conf_dir).resolve())):
                    # Inclure le noeud hydra et initialiser explicitement HydraConfig pour get_original_cwd()
                    cfg = compose(config_name="config", return_hydra_config=True)
                    HydraConfig.instance().set_config(cfg)
                    # Optionnel: retirer le noeud hydra de la config applicative
                    with open_dict(cfg):
                        if "hydra" in cfg:
                            del cfg["hydra"]
                    app = AppOrchestrator(cfg)
                    return app, cast(DictConfig, cfg), None
            except Exception as e:
                last_err = f"{e}"

    # 2) Fallback: simple lecture (si pas de defaults Hydra)
    for candidate in ("conf/config.yaml", "/app/conf/config.yaml"):
        try:
            p = Path(candidate)
            if p.is_file():
                cfg2 = cast(DictConfig, OmegaConf.load(str(p)))
                app2 = AppOrchestrator(cfg2)
                return app2, cfg2, None
        except Exception as e:
            last_err = f"{e}"

    # 3) Fallback minimal compatible AppOrchestrator (inclut orchestrators)
    try:
        cfg3 = cast(
            DictConfig,
            OmegaConf.create(
                {
                    "project": {
                        "name": os.getenv(ENV_PROJECT_NAME, "demo_project"),
                        "output_dir": os.getenv(ENV_OUTPUTS_DIR, "outputs"),
                        "random_state": 42,
                    },
                    "logger": {
                        "backend": "stdlib",
                        "app_name": os.getenv(ENV_PROJECT_NAME, "mlp"),
                        "level": "INFO",
                        "json_mode": False,
                    },
                    "orchestrators": {
                        "config": {"enabled": True},
                        "message": {
                            "enabled": True,
                            "locale": os.getenv(ENV_LANG, "fr"),
                            "locales_dir": DEFAULT_LOCALES_DIR,
                            "domains": [
                                "general",
                                "config",
                                "file",
                                "data",
                                "eda",
                                "pipeline",
                                "report",
                                "streamlit_app",
                            ],
                        },
                        "file": {"enabled": True, "data_dir": "data", "in_dir": "in", "out_dir": "out"},
                        "data": {"enabled": True},
                        "eda": {"enabled": True},
                        "pipeline": {"enabled": True},
                        "report": {"enabled": True, "formats": ["html", "md"]},
                    },
                    "i18n": {
                        "locales_dir": DEFAULT_LOCALES_DIR,
                        "domain": DEFAULT_I18N_DOMAIN,
                        "locale": os.getenv(ENV_LANG, "fr"),
                    },
                }
            ),
        )
        app3 = AppOrchestrator(cfg3)
        return app3, cfg3, None
    except Exception as e:
        last_err = f"{e}"

    return None, None, last_err or "Unknown error"


# -------------------- Defaults UI (session) --------------------
def _init_defaults(context: dict[str, str] | None, cfg: DictConfig) -> None:
    ss: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)
    outputs_root = (context or {}).get("outputs_root") or os.getenv(ENV_OUTPUTS_DIR, "outputs")
    project_name = (
        (cfg.get("project", {}) or {}).get("name", os.getenv(ENV_PROJECT_NAME, "demo_project"))
        if isinstance(cfg, Mapping)
        else os.getenv(ENV_PROJECT_NAME, "demo_project")
    )
    defaults: dict[str, Any] = {
        SS_OUTPUTS_DIR: outputs_root,
        SS_PROJECT_NAME: project_name,
        SS_NOTEBOOK_DIR: os.getenv(ENV_NOTEBOOK_DIR, "notebooks"),
        SS_NOTEBOOK_URL: os.getenv(ENV_NOTEBOOK_URL, ""),
        SS_LANG: (cfg.get("i18n", {}) or {}).get("locale", os.getenv(ENV_LANG, "fr"))
        if isinstance(cfg, Mapping)
        else os.getenv(ENV_LANG, "fr"),
        SS_DOCS_DIR: os.getenv(ENV_DOCS_DIR, "docs"),
        SS_LOG_FILE: os.getenv(ENV_LOG_FILE, "/logs/streamlit_app.log"),
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v


# -------------------- i18n (depuis AppOrchestrator) --------------------
def _install_translator(app: AppOrchestrator, cfg: DictConfig) -> Callable[[str, Any], str]:
    domain = (cfg.get("i18n", {}) or {}).get("domain", "streamlit_app")
    mo_app = getattr(app, "message_orchestrator", None)
    st.session_state["message_orchestrator"] = mo_app  # exposer pour les pages

    def tr_noop(key: str, **p: Any) -> str:
        return key

    if mo_app and hasattr(mo_app, "translate") and callable(getattr(mo_app, "translate")):
        def tr_translate(key: str, **p: Any) -> str:
            return mo_app.translate(domain, key, **p)  # type: ignore[attr-defined]
        st.session_state["tr"] = tr_translate
        return tr_translate

    st.session_state["tr"] = tr_noop
    return tr_noop


# -------------------- Images Markdown: CSS + Réécriture --------------------
_IMG_MD_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)')
_IMG_HTML_RE = re.compile(r'[^>]*src=["\'](?P<src>[^"\']+)["\'][^>]*>')

def _inject_md_image_css() -> None:
    st.markdown(
        """
        <style>
        /* Styles additionnels si besoin */
        .stMarkdown img { max-width: 100%; height: auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _rewrite_image_urls(md_text: str, md_file: str, docs_base: str = "docs") -> str:
    md_dir = Path(md_file).parent
    try:
        md_dir_rel = md_dir.relative_to(docs_base)  # ex: fr/home
    except Exception:
        return md_text
    docs_url_root = os.path.basename(os.path.normpath(docs_base)) if os.path.isabs(docs_base) else docs_base
    static_base = f"app/static/{docs_url_root}/{md_dir_rel.as_posix()}/"

    def _rewrite_src(src: str) -> str:
        s = src.strip().strip("'\"")
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

    md_text = _IMG_MD_RE.sub(md_sub, md_text)
    md_text = _IMG_HTML_RE.sub(html_sub, md_text)
    return md_text


# -------------------- Rendu Markdown (docs) --------------------
def render_docs(section: str, lang: str | None = None) -> None:
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


# -------------------- Préchargement File→Data (sans EDA/Pipeline/Report) --------------------
def _find_initial_dataset(context: dict[str, Any]) -> Path | None:
    # Priorité à dataset_V5.csv si présent, sinon premier fichier éligible
    data_in = Path(context.get("data_in", "data/in"))
    if not data_in.exists():
        return None
    preferred = data_in / "dataset_V5.csv"
    if preferred.is_file():
        return preferred
    for ext in ("*.csv", "*.xlsx", "*.xls", "*.json"):
        matches = sorted(data_in.glob(ext))
        if matches:
            return matches[0]
    return None

@log_call_ex("streamlit.preload_data")  # log start/end + error via logger_manager/structlog/logging
def _preload_file_to_data(context: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    path = _find_initial_dataset(context)
    if path is None:
        return {}
    data_cfg = {}
    if isinstance(cfg, Mapping):
        # Prendre la section orchestrators.data si disponible
        data_cfg = cast(dict[str, Any], (cfg.get("orchestrators", {}) or {}).get("data", {}) or {})
    dm = DataManager(config=data_cfg)
    X, y = dm.prepare_for_ml(path)
    meta = {
        "path": str(path),
        "n_rows": getattr(X, "shape", [None, None])[0],
        "n_cols": getattr(X, "shape", [None, None])[1],
        "y_present": y is not None,
    }
    return {"X": X, "y": y, "metadata": meta}


# -------------------- UI: registres et sidebar --------------------
def _pages_registry(tr: Callable[[str, Any], str]) -> "OrderedDict[str, Callable[[], None]]":
    return OrderedDict(
        [
            (tr("NAV_HOME") if callable(tr) else "Accueil", home.run),
            (tr("NAV_EDA") if callable(tr) else "EDA", eda.run),
            (tr("NAV_PIPELINE") if callable(tr) else "Pipelines", pipeline.run),
            (tr("NAV_REPORT") if callable(tr) else "Rapports", report.run),
            (tr("NAV_NOTEBOOK") if callable(tr) else "Notebooks", notebook.run),
            (tr("NAV_DEMO") if callable(tr) else "Démo", demo.run),
            (tr("NAV_LOGS") if callable(tr) else "Logs", logs.run),
        ]
    )

def _sidebar(tr: Callable[[str, Any], str], pages: "OrderedDict[str, Callable[[], None]]") -> str:
    with st.sidebar:
        st.header(tr("APP_TITLE") if callable(tr) else APP_TITLE)
        page_label = st.selectbox(
            label=tr("LBL_PAGE") if callable(tr) else "Page",
            options=list(pages.keys()),
            index=0,
        )
        st.text_input(
            label=tr("LBL_PROJECT") if callable(tr) else "Projet",
            value=cast(str, st.session_state.get(SS_PROJECT_NAME, "")),
            key=SS_PROJECT_NAME,
        )
        st.text_input(
            label=tr("LBL_OUTPUTS_DIR") if callable(tr) else "Outputs dir",
            value=cast(str, st.session_state.get(SS_OUTPUTS_DIR, "")),
            key=SS_OUTPUTS_DIR,
        )
        st.text_input(
            label=tr("LBL_DOCS_DIR") if callable(tr) else "Docs dir",
            value=cast(str, st.session_state.get(SS_DOCS_DIR, "")),
            key=SS_DOCS_DIR,
        )
        st.text_input(
            label=tr("LBL_NOTEBOOKS_DIR") if callable(tr) else "Notebooks dir",
            value=cast(str, st.session_state.get(SS_NOTEBOOK_DIR, "")),
            key=SS_NOTEBOOK_DIR,
        )
        st.text_input(
            label=tr("LBL_NOTEBOOKS_URL") if callable(tr) else "Notebooks URL",
            value=cast(str, st.session_state.get(SS_NOTEBOOK_URL, "")),
            key=SS_NOTEBOOK_URL,
        )
        st.text_input(
            label=tr("LBL_LANG") if callable(tr) else "Langue",
            value=cast(str, st.session_state.get(SS_LANG, "")),
            key=SS_LANG,
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button(tr("BTN_CLEAR_CACHE") if callable(tr) else "Vider le cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(tr("MSG_CACHE_CLEARED") if callable(tr) else "Cache vidé.")
        with col2:
            if st.button(tr("BTN_HELP") if callable(tr) else "Aide"):
                st.info(HELP_TEXT)
        return page_label


# -------------------- Entrée principale --------------------
def main() -> None:
    # 1) Construire AppOrchestrator (Hydra-first + fallback)
    app, cfg, err = _build_app_orchestrator()
    if app is None or cfg is None:
        st.error(f"AppOrchestrator init failed: {err}")
        return

    # 2) Installer le traducteur Streamlit à partir de l’orchestrateur Message
    tr = _install_translator(app, cfg)

    # 3) Contextes/chemins depuis AppOrchestrator pour préremplir l’UI
    context = getattr(app, "context", {}) if hasattr(app, "context") else {}
    _init_defaults(context, cfg)

    # 4) Exposer les ressources standardisées (alignées standalone)
    st.session_state[SS_RENDER_DOCS] = render_docs
    st.session_state[SS_LOGGER_MANAGER] = getattr(app, "logger_manager", None)
    st.session_state[SS_CTX] = context
    st.session_state[SS_APP_CONFIG] = cfg

    # 5) Préchargement File→Data (unique, sans EDA/Pipeline/Report)
    if SS_DATA_RESULT not in st.session_state:
        try:
            if isinstance(cfg, Mapping):
                orch = cast(dict[str, Any], cfg.get("orchestrators", {}) or {})
                if (orch.get("file", {}).get("enabled", True)) and (orch.get("data", {}).get("enabled", True)):
                    payload = _preload_file_to_data(context, cfg)
                    if payload:
                        st.session_state[SS_DATA_RESULT] = payload
        except Exception as e:
            st.warning(f"Préchargement File→Data indisponible: {e}")

    # 6) Navigation/pages
    pages = _pages_registry(tr)
    page_label = _sidebar(tr, pages)
    st.title(tr("APP_TITLE") if callable(tr) else APP_TITLE)
    runner = pages.get(page_label)
    if runner is None:
        st.error(f"Page inconnue: {page_label}")
        return
    runner()


if __name__ == "__main__":
    main()
