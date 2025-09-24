from __future__ import annotations

import gettext
from collections.abc import Callable
from pathlib import Path
from typing import Any

"""
MessageManager: charge et résout des messages localisés via gettext.

- Locales attendues sous i18n/locales/<lang>/LC_MESSAGES/<domain>.mo
- Fallback sur 'en' si le domaine/langue est manquant, sinon identité.
"""

class MessageManager:
    """Gestionnaire de traduction pour multiples domaines."""

    def __init__(self, locales_dir: Path | str, default_locale: str = "fr") -> None:
        self._locales_dir = Path(locales_dir)
        self._default_locale = default_locale
        self._fallback = self._build_fallback()

    def _build_fallback(self) -> Callable[[str], str]:
        try:
            t = gettext.translation(
                domain="streamlit_app",
                localedir=str(self._locales_dir),
                languages=["en"],
            )
            return t.gettext
        except Exception:
            return lambda s: s

    def translator(self, domain: str, locale: str | None = None) -> Callable[[str], str]:
        """Retourne une fonction gettext pour un domaine/locale."""
        loc = locale or self._default_locale
        try:
            t = gettext.translation(
                domain=domain,
                localedir=str(self._locales_dir),
                languages=[loc],
            )
            return t.gettext
        except Exception:
            return self._fallback

    def msg(self, domain: str, key: str, locale: str | None = None, **params: Any) -> str:
        """Résout key dans domain, applique format(**params) si fourni."""
        _ = self.translator(domain, locale)
        template = _(key)
        try:
            return template.format(**params) if params else template
        except Exception:
            return template
