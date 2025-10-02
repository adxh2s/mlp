from __future__ import annotations

"""
File orchestrator.

Rôle:
- Sélectionner un fichier d'entrée (data/in) selon une liste d'extensions autorisées.
- Optionnellement copier et compresser ce fichier dans data/out (traçabilité).
- Charger le contenu en mémoire (DataFrame/objet) via FileManager.
- Émettre des événements localisés (MessageOrchestratorApp) et des logs structurés.

Contrat:
- Le contexte (context) est fourni par ConfigOrchestrator/AppOrchestrator et doit contenir:
  - 'data_in': chemin absolu du dossier d'entrée.
  - 'data_out': chemin absolu du dossier de sortie.
- Aucun fallback de chemins (Hydra/cwd) n'est réalisé ici, pour préserver la séparation des responsabilités.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from src.config.schemas import FileConfig as FileConfigModel
from src.instrumentation.file_manager import FileManager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.message_taxonomy import (
    FILE_INIT,
    INPUT_FOUND,
    INPUT_PROCESSED,
    NO_INPUT_FILE,
)
from src.orchestrators.bootstrap import bootstrap_instance  # bootstrap cfg uniquement
from src.orchestrators.message import MessageOrchestratorApp  # alignement app-level

# =========================
# Constantes de module
# =========================
LOGGER_NAME = "mlp.orchestrators.file"
DOMAIN = "file"

EV_FILE_PATHS_RESOLVED = "file_paths_resolved"

LOG_EXTRA = "extra_fields"
FIELD_FILE = "file"
FIELD_IN_DIR = "in_dir"
FIELD_OUT_DIR = "out_dir"
FIELD_DATA_DIR = "data_dir"
FIELD_EXTENSIONS = "extensions"
FIELD_SAVED = "saved"
FIELD_COMPRESSED = "compressed"

CTX_DATA_IN = "data_in"
CTX_DATA_OUT = "data_out"

KEY_FOUND = "found"
KEY_FILE = "file"
KEY_SAVED = "saved_copy"
KEY_COMPRESSED = "saved_copy_compressed"
KEY_DATA = "data"
KEY_META = "meta"


@dataclass(slots=True)
class FileConfig:
    """
    Configuration interne normalisée pour l'orchestrateur de fichiers.

    Notes:
    - La config effective vient typiquement de Pydantic (FileConfigModel) ou d'un dict (Hydra);
      ce dataclass sert de réceptacle après normalisation.
    """
    enabled: bool = True
    data_dir: str = "data"
    in_dir: str = "in"
    out_dir: str = "out"
    extensions: list[str] = field(default_factory=lambda: [".csv", ".xlsx", ".json"])
    save_input_file: bool = True
    save_input_file_compression: bool = False
    preferred_filename: str | None = None


class FileConfigDict(TypedDict, total=False):
    """Type dict minimal pour la normalisation de configuration en entrée."""
    enabled: bool
    data_dir: str
    in_dir: str
    out_dir: str
    extensions: list[str]
    save_input_file: bool
    save_input_file_compression: bool
    preferred_filename: str | None


def _coerce_cfg_dict(raw: Mapping[str, Any]) -> FileConfigDict:
    """
    Normalise un mapping arbitraire en FileConfigDict.
    - Cast des types simples.
    - Filtrage des séquences pour extensions.
    """
    out: FileConfigDict = {}
    if "enabled" in raw:
        out["enabled"] = bool(raw["enabled"])
    if "data_dir" in raw:
        out["data_dir"] = str(raw["data_dir"])
    if "in_dir" in raw:
        out["in_dir"] = str(raw["in_dir"])
    if "out_dir" in raw:
        out["out_dir"] = str(raw["out_dir"])
    if "extensions" in raw:
        exts = raw["extensions"]
        if isinstance(exts, Sequence) and not isinstance(exts, (str, bytes)):
            out["extensions"] = [str(e) for e in exts]
    if "save_input_file" in raw:
        out["save_input_file"] = bool(raw["save_input_file"])
    if "save_input_file_compression" in raw:
        out["save_input_file_compression"] = bool(raw["save_input_file_compression"])
    if "preferred_filename" in raw:
        val = raw["preferred_filename"]
        out["preferred_filename"] = None if val is None else str(val)
    return out


def _to_dict_cfg(cfg: object) -> FileConfigDict:
    """
    Convertit une config Pydantic/dataclass/dict en FileConfigDict.
    - Utilise model_dump() si disponible (Pydantic v2).
    - asdict() pour dataclass.
    - Mapping direct sinon.
    """
    md = getattr(cfg, "model_dump", None)
    if callable(md):
        try:
            raw_any: Any = md()
            if isinstance(raw_any, Mapping):
                return _coerce_cfg_dict(raw_any)
        except Exception:
            pass
    if is_dataclass(cfg) and not isinstance(cfg, type):
        raw_dc = asdict(cfg)
        return _coerce_cfg_dict(cast(Mapping[str, Any], raw_dc))
    if isinstance(cfg, Mapping):
        return _coerce_cfg_dict(cfg)
    return {}


def _evt(e: object) -> str:
    """Uniformise un identifiant d’événement issu d’un Enum/tuple/str en str."""
    if isinstance(e, str):
        return e
    if isinstance(e, (list, tuple)) and len(e) > 0:
        return str(e[0])
    return str(e)


DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "data_dir": "data",
    "in_dir": "in",
    "out_dir": "out",
    "extensions": [".csv", ".xlsx", ".json"],
    "save_input_file": True,
    "save_input_file_compression": False,
    "preferred_filename": None,
}


class FileOrchestrator(LoggerMixin):
    """
    Orchestrateur de fichiers.

    Responsabilités:
    - Découvrir un fichier d'entrée dans data_in selon les extensions autorisées.
    - Copier/compresser (optionnel) vers data_out pour traçabilité.
    - Charger les données via FileManager et renvoyer un payload riche pour la suite.

    Dépendances:
    - cfg: configuration de fichiers (Pydantic/FileConfig/dict).
    - logger_manager: LoggerManager optionnel pour logs structurés.
    - context: contexte applicatif OBLIGATOIRE avec 'data_in' et 'data_out' (absolus).

    Important:
    - Aucune résolution de chemins via Hydra/cwd n’est réalisée ici; le context provient
      de ConfigOrchestrator/AppOrchestrator qui garantit des chemins valides.
    """

    def __init__(
        self,
        cfg: FileConfigModel | FileConfig | Mapping[str, Any],
        logger_manager: LoggerManager | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        raw = _to_dict_cfg(cfg)
        self.cfg = FileConfig(**raw) if raw else FileConfig()
        self.context = context or {}
        self.fm = FileManager()

        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager is not None:
            self._init_logger(cast(Any, logger_manager))
        else:
            self.log = logging.getLogger(LOGGER_NAME)

        self.msg: MessageOrchestratorApp | None = None

        # Contexte requis
        missing = [k for k in (CTX_DATA_IN, CTX_DATA_OUT) if k not in self.context]
        if missing:
            raise ValueError(
                f"FileOrchestrator requires context with keys {missing}; "
                "build context via ConfigOrchestrator.run() (or AppOrchestrator) and inject it."
            )

        self.in_dir = Path(self.context[CTX_DATA_IN]).resolve()
        self.out_dir = Path(self.context[CTX_DATA_OUT]).resolve()

        # in_dir requis; out_dir best-effort (tolérer RO / non créable)
        self.fm.ensure_dir(self.in_dir)
        try:
            self.fm.ensure_dir(self.out_dir)
        except (OSError, PermissionError) as e:
            try:
                self.log.warning(
                    "ensure_out_dir_failed",
                    extra={LOG_EXTRA: {FIELD_OUT_DIR: str(self.out_dir), "error": str(e)}},
                )
            except Exception:
                pass

    @classmethod
    def bootstrap(
        cls,
        *,
        context_provider,  # callable: name -> mapping (doit fournir context avec data_in/data_out)
        logger_manager: LoggerManager | None = None,
        message_orchestrator: MessageOrchestratorApp | None = None,
        ini_filenames: tuple[str, ...] = ("file.ini", "default.ini"),
    ) -> "FileOrchestrator":
        """
        Bootstrap de configuration uniquement (aucun fallback de chemins); context reste obligatoire.
        - Priorité: contexte applicatif -> INI -> defaults, puis validate à l'instanciation.
        """
        def factory(params: dict[str, Any]) -> "FileOrchestrator":
            context = params.pop("_context", {})
            inst = cls(params, logger_manager=logger_manager, context=context)
            if message_orchestrator is not None:
                inst.attach_message(message_orchestrator)
            return inst

        def validator(inst: "FileOrchestrator") -> None:
            return

        def wrapped_context_provider(_name: str) -> dict[str, Any] | None:
            context = context_provider("file") or {}
            params = dict(context.get("orchestrators", {}).get("file", {})) if isinstance(context.get("orchestrators"), dict) else {}
            params["_context"] = context
            return params

        return bootstrap_instance(
            name="file",
            factory=factory,
            defaults=DEFAULTS,
            validator=validator,
            context_provider=wrapped_context_provider,
            ini_filenames=ini_filenames,
        )

    @classmethod
    def from_cfg_mgr(
        cls,
        cfg_mgr: Any,
        logger_manager: LoggerManager | None = None,
        context: dict[str, str] | None = None,
    ) -> FileOrchestrator:
        """Fabrique un FileOrchestrator à partir d’un ConfigManager-like (.model.orchestrators.file attendu)."""
        cfg = cfg_mgr.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager, context=context)

    @classmethod
    def from_config_manager(
        cls,
        config_manager: Any,
        logger_manager: LoggerManager | None = None,
        context: dict[str, str] | None = None,
    ) -> FileOrchestrator:
        """Alias de from_cfg_mgr pour compatibilité d’API explicite."""
        cfg = config_manager.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager, context=context)

    def attach_message(self, msg: MessageOrchestratorApp) -> None:
        """Attache l’orchestrateur de messages pour émettre les événements localisés."""
        self.msg = msg

    def pick_input_file(self) -> Path | None:
        """
        Sélectionne le fichier d’entrée via FileManager:
        - Priorité à preferred_filename s’il existe et est présent.
        - Sinon, plus récent par mtime parmi les extensions autorisées.
        """
        return self.fm.pick_input_file(
            in_dir=self.in_dir,
            exts=self.cfg.extensions,
            preferred_filename=self.cfg.preferred_filename,
        )

    def process_input(self) -> dict[str, Any]:
        """
        Traite l’entrée fichier:
        - Émet FILE_INIT et file_paths_resolved.
        - Cherche un fichier; si absent, NO_INPUT_FILE et payload found=False.
        - Copie/compresse si configuré (tolérant RO), lit les données, émet INPUT_PROCESSED.
        - Retourne un dict avec clés: found, file, saved_copy, saved_copy_compressed, data, meta.
        """
        # Init event
        if self.msg:
            self.msg.emit(
                DOMAIN,
                _evt(FILE_INIT),
                **{FIELD_DATA_DIR: self.cfg.data_dir, FIELD_IN_DIR: self.cfg.in_dir, FIELD_OUT_DIR: self.cfg.out_dir},
            )
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                extra_payload: dict[str, Any] = {
                    LOG_EXTRA: {FIELD_DATA_DIR: self.cfg.data_dir, FIELD_IN_DIR: self.cfg.in_dir, FIELD_OUT_DIR: self.cfg.out_dir}
                }
                logger.info("file_init", extra=extra_payload)

        # Paths resolved
        if self.msg:
            self.msg.emit(DOMAIN, EV_FILE_PATHS_RESOLVED, **{FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir)})
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                extra_payload = {LOG_EXTRA: {FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir)}}
                logger.info(EV_FILE_PATHS_RESOLVED, extra=extra_payload)

        # Pick file via FileManager
        f = self.pick_input_file()
        if f is None:
            if self.msg:
                self.msg.emit(DOMAIN, _evt(NO_INPUT_FILE), **{FIELD_IN_DIR: str(self.in_dir), FIELD_EXTENSIONS: self.cfg.extensions})
            else:
                logger = getattr(self, "log", None)
                if logger is not None:
                    extra_payload = {LOG_EXTRA: {FIELD_IN_DIR: str(self.in_dir), FIELD_EXTENSIONS: list(self.cfg.extensions)}}
                    logger.info("no_input_file", extra=extra_payload)
            return {
                KEY_FOUND: False,
                KEY_FILE: None,
                KEY_SAVED: None,
                KEY_COMPRESSED: None,
                KEY_DATA: None,
                KEY_META: {FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir), FIELD_EXTENSIONS: self.cfg.extensions},
            }

        # Found
        if self.msg:
            self.msg.emit(DOMAIN, _evt(INPUT_FOUND), **{FIELD_FILE: str(f)})
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("input_found", extra={LOG_EXTRA: {FIELD_FILE: str(f)}})

        # Save and compress if configured (tolérant RO)
        saved_path: Path | None = None
        compressed_path: Path | None = None
        if self.cfg.save_input_file:
            try:
                stamped = self.fm.make_timestamp_name(f)  # accepte Path/str selon l’implémentation FileManager
                saved_path = self.fm.copy_file(f, self.out_dir, rename=stamped)
                if self.cfg.save_input_file_compression and saved_path is not None:
                    compressed_path = self.fm.compress_file_gz(saved_path, delete_original=True)
            except (OSError, PermissionError) as e:
                logger = getattr(self, "log", None)
                if logger is not None:
                    logger.warning(
                        "file_copy_skipped_rofs",
                        extra={LOG_EXTRA: {FIELD_OUT_DIR: str(self.out_dir), "error": str(e)}},
                    )
                saved_path, compressed_path = None, None

        # Read data via FileManager
        data = self.fm.read_file(f)

        # Done
        if self.msg:
            self.msg.emit(
                DOMAIN,
                _evt(INPUT_PROCESSED),
                **{
                    FIELD_FILE: str(f),
                    FIELD_SAVED: (str(saved_path) if saved_path is not None else None),
                    FIELD_COMPRESSED: (str(compressed_path) if compressed_path is not None else None),
                },
            )
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                extra_payload = {
                    LOG_EXTRA: {
                        FIELD_FILE: str(f),
                        FIELD_SAVED: (str(saved_path) if saved_path is not None else None),
                        FIELD_COMPRESSED: (str(compressed_path) if compressed_path is not None else None),
                    }
                }
                logger.info("input_processed", extra=extra_payload)

        return {
            KEY_FOUND: True,
            KEY_FILE: str(f),
            KEY_SAVED: (str(saved_path) if saved_path is not None else None),
            KEY_COMPRESSED: (str(compressed_path) if compressed_path is not None else None),
            KEY_DATA: data,
            KEY_META: {FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir), FIELD_EXTENSIONS: self.cfg.extensions},
        }
