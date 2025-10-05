import os
import re
import gettext
from pathlib import Path
from collections.abc import Callable

import streamlit as st
from src.instrumentation.decorators import log_page

"""
Page Logs Streamlit :
- Affichage des logs (pleine largeur, sous les filtres)
- Diagnostic i18n avancé avec sélection dynamique du domaine (.mo)
- Initialisation automatique du domaine i18n selon le nom du script principal (ici "streamlit_app")
Conforme PEP8, typage strict, maintenabilité assurée.
"""

# --------------------------------------------------------------------------
# Initialisation : fixe le domaine i18n par défaut selon le nom du script principal
# Si l'utilisateur change le domaine via selectbox, on met à jour la session
# --------------------------------------------------------------------------
DEFAULT_I18N_DOMAIN = Path(__file__).stem  # "streamlit_app" si script : streamlit_app.py
if "i18n_domain" not in st.session_state:
    st.session_state["i18n_domain"] = DEFAULT_I18N_DOMAIN

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
DEFAULT_LOG_FILE: str = os.getenv("MLP_LOG_FILE", "streamlit_app.log")
I18N_SAMPLE_KEYS: list[str] = [
    "APP_TITLE", "NAV_HOME", "NAV_EDA", "NAV_PIPELINE", "NAV_REPORT", "NAV_LOGS", "BTN_HELP"
]
LOG_TEXTAREA_HEIGHT: int = 900

# --------------------------------------------------------------------------
# Fonction utilitaire pour lire la fin du fichier log (côté serveur)
# --------------------------------------------------------------------------
def _read_tail(path: str, max_lines: int) -> list[str]:
    """
    Lit les dernières lignes du fichier log en gérant les erreurs Unicode

    Args:
        path: Chemin complet du fichier log à lire
        max_lines: Nombre de lignes à retourner à partir de la fin

    Returns:
        List[str]: Liste des lignes lues ou message d'erreur
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-max_lines:] if max_lines > 0 else lines
    except Exception as exc:
        return [f"[logs] Impossible de lire '{path}': {exc}\n"]

@log_page("logs")
def run() -> None:
    """
    Entrée principale de la page Streamlit 'Logs'.
    - Affiche les logs avec filtres sur toute la largeur
    - Permet un diagnostic i18n avancé et la sélection dynamique du domaine
    - Domaine par défaut fixé sur streamlit_app au lancement
    """

    # ----------------------------------------------------------------------
    # Header et filtres logs (input de sélection, regex, nb lignes)
    # ----------------------------------------------------------------------
    tr: Callable[[str], str] = st.session_state.get("tr", lambda s, **p: s)
    st.header(tr("NAV_LOGS", None) if callable(tr) else "Logs")

    log_file: str = st.text_input("Fichier log", value=st.session_state.get("log_file", DEFAULT_LOG_FILE))
    st.session_state["log_file"] = log_file
    max_lines: int = st.number_input("Dernières lignes", min_value=50, max_value=10000, step=50, value=500)
    pattern: str = st.text_input("Filtre regex", value="")

    # Lecture et filtrage des logs puis affichage sur toute la largeur
    lines: list[str] = _read_tail(log_file, max_lines)
    if pattern:
        try:
            lines = [l for l in lines if re.search(pattern, l)]
        except Exception as exc:
            st.warning(f"Pattern invalide : {exc}")
    st.text_area("Logs", value="".join(lines), height=LOG_TEXTAREA_HEIGHT)

    # ----------------------------------------------------------------------
    # Bloc de diagnostic i18n
    # - Liste dynamique des domaines trouvés sous i18n/locales/lang/LC_MESSAGES/*.mo
    # - Charge automatiquement streamlit_app en première position si dispo
    # - Persiste la sélection dans session_state
    # ----------------------------------------------------------------------
    with st.expander("Test i18n streamlit_app", expanded=False):
        localedir = st.session_state.get("localedir", "i18n/locales")
        lang = st.session_state.get("lang", "fr")
        mo_dir = Path(localedir) / lang / "LC_MESSAGES"

        # Détection des domaines : tous les fichiers .mo présents pour le selectbox
        mo_files = sorted([p.stem for p in mo_dir.glob("*.mo")])
        # streamlit_app prioritaire
        sel_domain_default = (mo_files.index("streamlit_app") if "streamlit_app" in mo_files else 0)
        domain = st.selectbox(
            "Domaine i18n à charger",
            options=mo_files,
            index=sel_domain_default
        )
        # Mémorisation du choix dans la session (pour l'affichage)
        st.session_state["i18n_domain"] = domain

        mo_path = mo_dir / f"{domain}.mo"
        st.write(f"Chemin .mo sélectionné : {mo_path}")

        if st.button("Charger catalogue i18n"):
            try:
                tr_get = gettext.translation(domain=domain, localedir=localedir, languages=[lang], fallback=False)
                catalog = getattr(tr_get, "_catalog", {}) or {}
                size = len([k for k in catalog.keys() if isinstance(k, str)])
                st.success(f"Catalogue chargé ({size} entrées)")
                st.table({
                    "msgid": I18N_SAMPLE_KEYS,
                    "msgstr": [tr_get.gettext(k) for k in I18N_SAMPLE_KEYS]
                })
            except Exception as exc:
                st.error(f"Erreur de chargement .mo : {exc}")
                st.info("Vérifiez domain/lang/localedir et compilez .po avec msgfmt.")

        # Preview des valeurs courantes dans message_orchestrator si présent
        mo = st.session_state.get("message_orchestrator")
        mo_core = getattr(mo, "core", mo)
        if mo_core:
            st.code({k: mo_core.get(k) for k in I18N_SAMPLE_KEYS}, language="json")

# Fin du script
