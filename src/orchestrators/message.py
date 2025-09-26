# src/orchestrators/message.py

from __future__ import annotations

import gettext
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

# =========================
# Core (faible couplage)
# =========================

class MessageOrchestrator:
    """
    Noyau i18n minimal basé sur gettext:
    - Paramétré par localedir / domain / default_lang
    - Expose get(key, params) et load(lang)
    - Fournit translate(domain, key, **params) en secours pour compatibilité
    """

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
        """Charge l’anglais comme secours s’il existe, sinon identité."""
        try:
            t = gettext.translation(self.domain, localedir=self.localedir, languages=["en"])
            self._fallback = t.gettext
        except Exception:
            self._fallback = lambda s: s

    def load(self, lang: str) -> None:
        """Charge/active la langue courante pour le domaine par défaut."""
        self.lang = lang
        try:
            t = gettext.translation(self.domain, localedir=self.localedir, languages=[self.lang])
            self._translator = t.gettext
        except Exception:
            self._translator = self._fallback

    def set_lang(self, lang: str) -> None:
        self.load(lang)

    def get(self, key: str, params: dict[str, Any] | None = None) -> str:
        """Traduit key dans le domaine courant, puis applique format(**params) si fourni."""
        template = self._translator(key)
        if params:
            try:
                return template.format(**params)
            except Exception:
                return template
        return template

    # Compatibilité: traduction sur domaine arbitraire
    def translate(self, domain: str, key: str, **params: Any) -> str:
        """Traduit une clé dans un domaine donné, avec fallback si absent."""
        try:
            if domain == self.domain:
                template = self._translator(key)
            else:
                t = gettext.translation(domain, localedir=self.localedir, languages=[self.lang])
                template = t.gettext(key)
        except Exception:
            template = self._fallback(key)
        if params:
            try:
                return template.format(**params)
            except Exception:
                return template
        return template


# =========================
# Wrapper “app-level”
# =========================

class MessageOrchestratorApp:
    """
    Orchestrateur i18n applicatif:
    - Signature homogène avec les autres orchestrateurs (config_manager, logger_manager)
    - Délègue la traduction au core (MessageOrchestrator)
    - Expose get(...), translate(...), emit(...) pour la journalisation structurée
    """

    def __init__(
        self,
        config_manager: Any,
        logger_manager: Optional[Any] = None,
        localedir: Optional[str | Path] = None,
        domain: Optional[str] = None,
        default_lang: Optional[str] = None,
    ) -> None:
        cfg = getattr(config_manager, "model", getattr(config_manager, "cfg", None))
        # Paramètres par défaut sûrs; possibilité de lire orchestrators.message.* depuis cfg si existant
        loc_dir = str(localedir or "i18n/locales")
        dom = domain or "streamlit_app"
        lang = default_lang or "fr"
        self.core = MessageOrchestrator(localedir=loc_dir, domain=dom, default_lang=lang)
        self.lm = logger_manager  # Optionnel; utilisé par emit(...)

    def set_lang(self, lang: str) -> None:
        self.core.set_lang(lang)

    def get(self, key: str, params: dict[str, Any] | None = None) -> str:
        return self.core.get(key, params)

    def translate(self, domain: str, key: str, **params: Any) -> str:
        return self.core.translate(domain, key, **params)

    def emit(self, domain: str, key: str, level: str = "info", **fields: Any) -> None:
        """
        Formate un message traduit et l’émet au logger si disponible.
        - level: "debug" | "info" | "warning" | "error" | "critical"
        - fields: champs structurés additionnels
        """
        msg = self.translate(domain, key, **fields)
        try:
            # LoggerManager compatible: si une méthode log(...) existe, on l’utilise
            if self.lm and hasattr(self.lm, "log"):
                self.lm.log(level=level, event=msg, **fields)
            # Sinon, tenter un logger standard si présent
            elif self.lm and hasattr(self.lm, "get_logger"):
                logger = self.lm.get_logger("mlp.i18n")
                getattr(logger, level, logger.info)(msg, extra=fields)  # type: ignore[attr-defined]
        except Exception:
            # Sécurité: ne jamais casser l’exécution si la journalisation échoue
            pass
