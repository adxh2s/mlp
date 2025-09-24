# src/orchestrators/messages.py
from __future__ import annotations

import gettext
from collections.abc import Callable
from pathlib import Path

"""
MessagesOrchestrator: service i18n basé sur gettext pour l'UI Streamlit.

- Charge les catalogues .mo sous i18n/locales/<lang>/LC_MESSAGES/streamlit_app.mo
- Expose tr(msgid: str, **params) -> str
- Fallback sur 'en' si la clé ou la langue sont absentes
"""


class MessagesOrchestrator:
    """Orchestrateur i18n encapsulant gettext pour l'UI."""

    def __init__(
        self,
        localedir: str | Path = "i18n/locales",
        domain: str = "streamlit_app",
        default_lang: str = "fr",
    ) -> None:
        self.localedir = str(localedir)
        self.domain = domain
        self.lang = default_lang
        self._translator: Callable[[str], str] = lambda s: s
        self._fallback: Callable[[str], str] = lambda s: s
        self._load_fallback()

    def _load_fallback(self) -> None:
        """Charge l'anglais comme secours s'il existe, sinon identité."""
        try:
            t = gettext.translation(self.domain, localedir=self.localedir, languages=["en"])
            self._fallback = t.gettext
        except Exception:
            self._fallback = lambda s: s  # identity

    def load(self, lang: str) -> None:
        """Charge la langue demandée et prépare la fonction de traduction."""
        self.lang = lang
        try:
            t = gettext.translation(self.domain, localedir=self.localedir, languages=[lang], fallback=False)
            self._translator = t.gettext
        except Exception:
            # si introuvable: repli sur fallback (anglais ou identité)
            self._translator = self._fallback

    def get(self, key: str, params: dict | None = None) -> str:
        """Traduit la clé et applique un formatage par paramètres nommés."""
        txt = self._translator(key)
        if params:
            try:
                return txt.format(**params)
            except Exception:
                # si mauvais mapping: renvoyer brut pour éviter les plantages UI
                return txt
        return txt
