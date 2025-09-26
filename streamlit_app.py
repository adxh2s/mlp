from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, cast

import streamlit as st
from omegaconf import DictConfig, OmegaConf

# Configuration de la page (unique)
APP_TITLE: str = "MLP App"
APP_ICON: str = "📊"
PAGE_LAYOUT: str = "wide"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)

# Imports dépendants du projet (tolérance si absents)
try:
    from src.instrumentation.config_manager import ConfigManager
except Exception:
    ConfigManager = None  # type: ignore[assignment]

try:
    from src.orchestrators.message import MessageOrchestrator
except Exception:
    MessageOrchestrator = None  # type: ignore[assignment]

# Pages (doivent exposer une fonction run())
from streamlit_pages import demo, eda, home, notebook, pipeline, report

# Texte d’aide affichable à la demande
HELP_TEXT = """
Entrée principale Streamlit: i18n via gettext, navigation multipages, config robuste.
- set_page_config appelé une seule fois.
- i18n centralisée (MessageOrchestrator) → st.session_state['tr'].
- Tolérance à l’absence de conf/config.yaml (config minimale).
"""

# Clés session_state
SS_OUTPUTS_DIR: str = "outputs_dir"
SS_PROJECT_NAME: str = "project_name"
SS_NOTEBOOK_DIR: str = "notebooks_dir"
SS_NOTEBOOK_URL: str = "notebooks_url"
SS_LANG: str = "lang"
SS_DOCS_DIR: str = "docs_dir"

# Variables d’environnement
ENV_OUTPUTS_DIR: str = "MLP_OUTPUTS_DIR"
ENV_PROJECT_NAME: str = "MLP_PROJECT_NAME"
ENV_NOTEBOOK_DIR: str = "MLP_NOTEBOOKS_DIR"
ENV_NOTEBOOK_URL: str = "MLP_NOTEBOOKS_URL"
ENV_LANG: str = "MLP_LANG"
ENV_DOCS_DIR: str = "MLP_DOCS_DIR"

# Defaults i18n
DEFAULT_I18N_DOMAIN: str = "streamlit_app"
DEFAULT_LOCALES_DIR: str = "i18n/locales"

# -------------------- Initialisation --------------------

def _init_defaults() -> None:
    ss: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)
    defaults: dict[str, Any] = {
        SS_OUTPUTS_DIR: os.getenv(ENV_OUTPUTS_DIR, "outputs"),
        SS_PROJECT_NAME: os.getenv(ENV_PROJECT_NAME, "demo_project"),
        SS_NOTEBOOK_DIR: os.getenv(ENV_NOTEBOOK_DIR, "notebooks"),
        SS_NOTEBOOK_URL: os.getenv(ENV_NOTEBOOK_URL, ""),
        SS_LANG: os.getenv(ENV_LANG, "fr"),
        SS_DOCS_DIR: os.getenv(ENV_DOCS_DIR, "docs"),
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

def _load_config() -> DictConfig:
    # Charge la config si disponible, sinon fallback minimal
    try:
        if ConfigManager is not None:
            cfg = ConfigManager.load_base_config()
            return cast(DictConfig, cfg)
    except Exception:
        pass
    return OmegaConf.create(
        {
            "project": {
                "name": st.session_state.get(SS_PROJECT_NAME, "demo_project"),
                "output_dir": st.session_state.get(SS_OUTPUTS_DIR, "outputs"),
            },
            "i18n": {
                "locales_dir": DEFAULT_LOCALES_DIR,
                "domain": DEFAULT_I18N_DOMAIN,
                "locale": st.session_state.get(SS_LANG, "fr"),
            },
        }
    )

def _init_i18n(cfg: DictConfig) -> Callable[[str], str]:
    tr: Callable[[str], str]
    try:
        if MessageOrchestrator is not None:
            locales_dir = (
                cfg.get("i18n", {}).get("locales_dir", DEFAULT_LOCALES_DIR)  # type: ignore[assignment]
                if isinstance(cfg, Mapping) else DEFAULT_LOCALES_DIR
            )
            domain = (
                cfg.get("i18n", {}).get("domain", DEFAULT_I18N_DOMAIN)  # type: ignore[assignment]
                if isinstance(cfg, Mapping) else DEFAULT_I18N_DOMAIN
            )
            locale = (
                cfg.get("i18n", {}).get("locale", "fr")  # type: ignore[assignment]
                if isinstance(cfg, Mapping) else "fr"
            )
            mo = MessageOrchestrator(
                locales_dir=locales_dir,
                domains=[domain],
                locale=locale,
                enabled=True,
            )
            tr = mo.tr
        else:
            tr = lambda s, **p: s
    except Exception:
        tr = lambda s, **p: s
    st.session_state["tr"] = tr
    return tr

# -------------------- Images Markdown: CSS + Réécriture --------------------

def _inject_md_image_css() -> None:
    # Contraint toute image Markdown à la largeur de la colonne (max-width:100%)
    st.markdown(
        "<style>.stMarkdown img{max-width:100%;height:auto;}</style>",
        unsafe_allow_html=True,
    )

IMG_MD_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)')
IMG_HTML_RE = re.compile(r'<img\s+[^>]*src=["\'](?P<src>[^"\']+)["\'][^>]*>')

def _rewrite_image_urls(md_text: str, md_file: str, docs_base: str = "docs") -> str:
    """
    Réécrit:
      images/xxx.png → app/static/docs/{lang}/{section}/images/xxx.png
      ./images/xxx.png → app/static/docs/{lang}/{section}/images/xxx.png
    Laisse intacts: http(s)://… et app/static/…
    """
    md_dir = Path(md_file).parent
    md_dir_rel = md_dir.relative_to(docs_base)  # ex: fr/home
    static_base = f"app/static/{docs_base}/{md_dir_rel.as_posix()}/"

    def _rewrite_src(src: str) -> str:
        s = src.strip().strip('\'"')
        if s.startswith(("http://", "https://", "app/static/")):
            return src
        if s.startswith(("./images/", "images/")):
            s2 = s.lstrip("./")
            return f"{static_base}{s2}"
        return src  # autres chemins: laisser tel quel

    def md_sub(m: re.Match) -> str:
        alt = m.group("alt")
        src = m.group("src")
        # Gérer optionnellement un "title" après le chemin
        if " " in src and not src.strip().startswith(("http://","https://","app/static/")):
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

# -------------------- Rendu Markdown par section --------------------

def render_docs(section: str, lang: str | None = None) -> None:
    """
    Recherche et affiche les Markdown d'une section selon la langue courante.
    Ordre de recherche:
      docs/{lang}/{section}/*.md
      docs/{section}.{lang}.*.md
      docs/{section}.*.{lang}.md
      docs/{section}.*.md
      docs/{section}/*.md
    Réécrit les liens d'images relatives vers app/static/docs/... pour service statique.
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

# -------------------- UI: registres et sidebar --------------------

def _pages_registry(tr: Callable[[str], str]) -> "OrderedDict[str, Callable[[], None]]":
    return OrderedDict(
        [
            (tr("NAV_HOME") if callable(tr) else "Accueil", home.run),
            (tr("NAV_EDA") if callable(tr) else "EDA", eda.run),
            (tr("NAV_PIPELINE") if callable(tr) else "Pipelines", pipeline.run),
            (tr("NAV_REPORT") if callable(tr) else "Rapports", report.run),
            (tr("NAV_NOTEBOOK") if callable(tr) else "Notebooks", notebook.run),
            (tr("NAV_DEMO") if callable(tr) else "Démo", demo.run),
        ]
    )

def _sidebar(tr: Callable[[str], str], pages: "OrderedDict[str, Callable[[], None]]") -> str:
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
    _init_defaults()
    cfg = _load_config()
    tr = _init_i18n(cfg)

    # Exposer le helper pour les pages (si besoin)
    st.session_state["render_docs"] = render_docs

    pages = _pages_registry(tr)
    page_label = _sidebar(tr, pages)

    st.title(APP_TITLE)

    runner = pages.get(page_label)
    if runner is None:
        st.error(f"Page inconnue: {page_label}")
        return
    runner()

if __name__ == "__main__":
    main()
