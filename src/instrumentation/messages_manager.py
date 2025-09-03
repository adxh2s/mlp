from __future__ import annotations

import gettext
from pathlib import Path
from typing import Any, Callable


class MessageManager:
    """Charge et résout des messages localisés par domaine (gettext)."""

    def __init__(self, locales_dir: Path, default_locale: str = "fr") -> None:
        self._locales_dir = locales_dir
        self._default_locale = default_locale

    def translator(self, domain: str, locale: str | None = None) -> Callable[[str], str]:
        loc = locale or self._default_locale
        try:
            t = gettext.translation(domain=domain, localedir=str(self._locales_dir), languages=[loc])
            return t.gettext
        except Exception:
            return lambda s: s

    def msg(self, domain: str, key: str, locale: str | None = None, **params: Any) -> str:
        _ = self.translator(domain, locale)
        template = _(key)
        try:
            return template.format(**params)
        except Exception:
            return template
