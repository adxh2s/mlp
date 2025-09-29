# src/orchestrators/message.py
from __future__ import annotations

import gettext
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.orchestrators.bootstrap import bootstrap_instance

DEFAULT_LOCALES_DIR = "i18n/locales"
DEFAULT_DOMAIN = "message"
DEFAULT_LANG = "fr"
VALIDATION_KEY = "NAV_HOME"  # Clé qui doit exister dans les .mo de déploiement

class MessageOrchestrator:
    """
    Noyau i18n minimal, autonome, basé sur gettext.
    - Paramétrage: localedir / domain / default_lang.
    - API: set_lang(lang), get(msgid, **params), translate(domain,msgid,**params).
    """

    def __init__(
        self,
        localedir: str | Path = DEFAULT_LOCALES_DIR,
        domain: str = DEFAULT_DOMAIN,
        default_lang: str = DEFAULT_LANG,
    ) -> None:
        self.localedir = str(localedir)
        self.domain = domain
        self.default_lang = default_lang
        self._translations: dict[str, gettext.GNUTranslations] = {}
        self._cur_lang = default_lang
        self._ensure_lang(default_lang)

    @classmethod
    def bootstrap(
        cls,
        *,
        context_provider,
        ini_filenames: tuple[str, ...] = ("message.ini", "default.ini"),
    ) -> "MessageOrchestrator":
        defaults = {"localedir": DEFAULT_LOCALES_DIR, "domain": DEFAULT_DOMAIN, "default_lang": DEFAULT_LANG}

        def factory(params: dict[str, Any]) -> "MessageOrchestrator":
            return cls(**params)

        def validator(inst: "MessageOrchestrator") -> None:
            # Vérifie que la clé de validation se traduit (ou retombe au msgid) sans lever
            _ = inst.get(VALIDATION_KEY)

        return bootstrap_instance(
            name="message",
            factory=factory,
            defaults=defaults,
            validator=validator,
            context_provider=context_provider,
            ini_filenames=ini_filenames,
        )

    def _ensure_lang(self, lang: str) -> None:
        if lang in self._translations:
            return
        try:
            trans = gettext.translation(self.domain, localedir=self.localedir, languages=[lang])
        except Exception:
            trans = gettext.NullTranslations()
        self._translations[lang] = trans
        if self._cur_lang != lang:
            self._cur_lang = lang

    def set_lang(self, lang: str) -> None:
        self._ensure_lang(lang)
        self._cur_lang = lang

    def get(self, msgid: str, **params: Any) -> str:
        tr = self._translations.get(self._cur_lang) or gettext.NullTranslations()
        try:
            text = tr.gettext(msgid)
            return text.format(**params) if params else text
        except Exception:
            return msgid

    def translate(self, domain: str, msgid: str, **params: Any) -> str:
        # Version minimale: un seul domain actif; si besoins multi-domaines, charger dynamiquement ici
        return self.get(msgid, **params)

class MessageOrchestratorApp:
    """
    Wrapper applicatif: délègue au core et prépare l’API emit(domain, code, **payload).
    """

    def __init__(self, core: MessageOrchestrator, domain_resolver: Callable[[str], str] | None = None) -> None:
        self.core = core
        self._domain_resolver = domain_resolver or (lambda d: d)

    def emit(self, domain: str, code: str, **payload: Any) -> None:
        _ = self.core.translate(self._domain_resolver(domain), code, **payload)
        return
