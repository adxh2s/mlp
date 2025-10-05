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
    - Paramétrage : localedir / domain / default_lang.
    - API : set_lang(lang), get(msgid, **params), translate(domain, msgid, **params)
    - Support natif multi-domaine, fallback automatique.

    Les fichiers de traduction doivent respecter la structure standard :
    i18n/locales/{lang}/LC_MESSAGES/{domain}.mo
    """

    def __init__(
        self,
        localedir: str | Path = DEFAULT_LOCALES_DIR,
        domain: str = DEFAULT_DOMAIN,
        default_lang: str = DEFAULT_LANG,
    ) -> None:
        """
        Initialise l’orchestrateur i18n : chemins, domaine principal, langue par défaut.
        """
        self.localedir = str(localedir)
        self.domain = domain
        self.default_lang = default_lang
        self._translations: dict[str, gettext.GNUTranslations] = {}  # Cache de translations Gettext
        self._cur_lang = default_lang
        self._ensure_lang(default_lang)

    @classmethod
    def bootstrap(
        cls,
        *,
        context_provider,
        ini_filenames: tuple[str, ...] = ("message.ini", "default.ini"),
    ) -> "MessageOrchestrator":
        """
        Méthode d’initialisation standard via bootstrap : supporte configuration Hydratée.
        Vérifie qu’une clé de validation est disponible à l’initialisation.
        """
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
        """
        S’assure que la langue spécifiée est bien chargée en cache.
        Si absent, charge une instance Gettext ou NullTranslations.
        """
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
        """
        Change la langue courante.
        """
        self._ensure_lang(lang)
        self._cur_lang = lang

    def get(self, msgid: str, **params: Any) -> str:
        """
        Traduit un message selon la langue courante, sur le domaine principal.
        Peut paramétrer la chaîne via .format(**params).
        """
        tr = self._translations.get(self._cur_lang) or gettext.NullTranslations()
        try:
            text = tr.gettext(msgid)
            return text.format(**params) if params else text
        except Exception:
            return msgid

    def translate(self, domain: str | list[str] | tuple[str, ...] = None, msgid: str = "", **params: Any) -> str:
        """
        Traduction multi-domaine avancée.

        - domain : chaîne, liste ou tuple, optionnel.
            - Si None, le domaine par défaut de la classe (`self.domain`) est utilisé automatiquement.
            - Si str: utilisé comme unique domaine (+ fallback sur self.domain si différent).
            - Si list/tuple : la liste fournie est parcourue (ordre donné), puis
              le domaine par défaut est ajouté en fin s’il n’est pas déjà présent.

        - msgid : clé à traduire (nom technique dans les .mo/.po).
        - params : dictionnaire des paramètres pour .format sur la chaîne traduite.

        Fallback : parcours chaque domaine et retourne la première traduction trouvée.
        Si aucune traduction trouvée, retourne le msgid original.

        Returns
        -------
        str
            Texte traduit, avec paramétrisation, ou clé originale si aucune traduction trouvée.
        """
        # Construction de la liste des domaines à interroger
        if domain is None:
            domains = [self.domain]
        elif isinstance(domain, str):
            domains = [domain]
            if domain != self.domain:
                domains.append(self.domain)
        else:
            domains = list(domain)
            if self.domain not in domains:
                domains.append(self.domain)
        # Parcours de chaque domaine en priorité donnée (ordre)
        for dom in domains:
            try:
                trans = gettext.translation(dom, localedir=self.localedir, languages=[self._cur_lang])
                text = trans.gettext(msgid)
                if text != msgid:
                    return text.format(**params) if params else text
            except Exception:
                continue
        return msgid

class MessageOrchestratorApp:
    """
    Wrapper applicatif pour l’orchestrateur i18n.  
    Permet de spécifier dynamiquement le domaine utilisé via le resolver.  
    Propose : emit(domain, code, **payload) – adapte à la logique multi-domaine.

    Paramètres
    ----------
    core : MessageOrchestrator
        Instance centrale de l’orchestrateur i18n.
    domain_resolver : Callable[[str], str | list[str] | tuple[str, ...]], optionnel
        Fonction pour adapter le domaine fourni à la signature multi-domaine : renvoie le domaine (str ou liste).
    """

    def __init__(
        self, core: MessageOrchestrator, domain_resolver: Callable[[str], str | list[str] | tuple[str, ...]] | None = None
    ) -> None:
        self.core = core
        # Le resolver doit retourner un domaine adapté à la fonction translate (str ou list)
        self._domain_resolver = domain_resolver or (lambda d: d)

    def emit(self, domain: str | list[str] | tuple[str, ...], code: str, **payload: Any) -> None:
        """
        Émet la traduction pour la clé code dans le(s) domaine(s) spécifié(s).
        La logique du domain_resolver permet de transformer dynamiquement le type de domaine.
        Compatible multi-domaine comme la méthode translate du core.
        
        Example : emit(["report", "nav"], "BTN_RUN_REPORT", param1="val1")
        """
        _ = self.core.translate(self._domain_resolver(domain), code, **payload)
        return
