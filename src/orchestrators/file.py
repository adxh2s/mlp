from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from hydra.utils import get_original_cwd

from src.config.schemas import FileConfig as FileConfigModel
from src.instrumentation.file_manager import FileManager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import (
    FILE_INIT,
    INPUT_FOUND,
    INPUT_PROCESSED,
    NO_INPUT_FILE,
)
from src.orchestrators.messages import MessageOrchestrator

"""
File orchestrator.
- Rôle: localiser un fichier d'entrée, l'optionnel copier/compresser pour traçabilité, le lire, et émettre des événements structurés. 
- Contexte: utilise ctx["data_in"]/ctx["data_out"] si fournis pour les chemins de travail, sinon retombe sur un ancrage Hydra (get_original_cwd). 
- Journalisation: compatible LoggerManager via LoggerMixin, et peut émettre des messages localisées via MessageOrchestrator.
"""

# =========================
# Constantes de module
# =========================

# Logger & domaine
LOGGER_NAME = "mlp.orchestrators.file"
DOMAIN = "file"

# Événements (internes, hors taxonomy)
EV_FILE_PATHS_RESOLVED = "file_paths_resolved"

# Champs de payload de logs
LOG_EXTRA = "extra_fields"
FIELD_FILE = "file"
FIELD_IN_DIR = "in_dir"
FIELD_OUT_DIR = "out_dir"
FIELD_DATA_DIR = "data_dir"
FIELD_EXTENSIONS = "extensions"
FIELD_SAVED = "saved"
FIELD_COMPRESSED = "compressed"

# Clés de contexte
CTX_DATA_IN = "data_in"
CTX_DATA_OUT = "data_out"

# Clés de retour
KEY_FOUND = "found"
KEY_FILE = "file"
KEY_SAVED = "saved_copy"
KEY_COMPRESSED = "saved_copy_compressed"
KEY_DATA = "data"
KEY_META = "meta"


@dataclass(slots=True)
class FileConfig:
    """
    Configuration opérationnelle du FileOrchestrator.
    - extensions: liste d'extensions recherchées pour le fichier d'entrée.
    - save_input_file: si True, copie du fichier source dans le répertoire de sortie avec horodatage.
    - save_input_file_compression: si True, compression .gz de la copie et suppression de l'original copié.
    - preferred_filename: nom préféré du fichier si présent dans data_in.
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
    """Projection typée de la configuration (clé/valeur) employée pour hydrater FileConfig."""
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
    Convertit un mapping arbitraire en FileConfigDict strictement typé.
    Garde-fous:
    - Forçage des bool/str.
    - Validation de extensions en list[str].
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
    Normalise toute forme de configuration (pydantic v2, dataclass instance, mapping)
    en FileConfigDict sans conversions risquées via dict(...) sur objets non-Mapping.
    Remarques:
    - Appelle .model_dump() uniquement si présent et callable.
    - Ne tente pas d'accéder à des attributs inconnus; sinon, renvoie {}.
    """
    # 1) Pydantic v2
    md = getattr(cfg, "model_dump", None)
    if callable(md):
        try:
            raw_any: Any = md()
            if isinstance(raw_any, Mapping):
                return _coerce_cfg_dict(raw_any)
        except Exception:
            pass  # fallback aux autres formes

    # 2) Dataclass instance
    if is_dataclass(cfg) and not isinstance(cfg, type):
        raw_dc = asdict(cfg)
        return _coerce_cfg_dict(cast(Mapping[str, Any], raw_dc))

    # 3) Mapping direct
    if isinstance(cfg, Mapping):
        return _coerce_cfg_dict(cfg)

    # 4) Rien de supporté
    return {}


def _evt(e: object) -> str:
    """
    Convertit une constante taxonomy (pouvant être tuple/list/str) en clé d'événement str.
    - Préserve la compatibilité avec les constantes (ex: ("file_init", ...)).
    - Évite iter/next génériques pour réduire les diagnostics Pylance.
    """
    if isinstance(e, str):
        return e
    if isinstance(e, (list, tuple)) and len(e) > 0:
        return str(e[0])
    return str(e)


class FileOrchestrator(LoggerMixin):
    """
    Orchestrateur de fichiers: détermine le fichier d'entrée à utiliser, gère la persistance/
    compression optionnelle, lit le contenu et émet des événements structurés pour observabilité.
    """

    def __init__(
        self,
        cfg: FileConfigModel | FileConfig | Mapping[str, Any],
        logger_manager: LoggerManager | None = None,
        ctx: dict[str, str] | None = None,
    ) -> None:
        """
        Initialise l'orchestrateur.
        - cfg: configuration pydantic/dataclass/dict; convertie en FileConfig robuste.
        - logger_manager: gestionnaire de log partagé; optionnel.
        - ctx: contexte chemins in/out (prioritaire sur la config).
        """
        raw = _to_dict_cfg(cfg)
        self.cfg = FileConfig(**raw) if raw else FileConfig()
        self.ctx = ctx or {}
        self.fm = FileManager()

        # Logger
        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager is not None:
            # LoggerMixin attend un objet avec get_logger(name: str|None) -> logging.Logger.
            self._init_logger(cast(Any, logger_manager))
        else:
            self.log = logging.getLogger(LOGGER_NAME)
        self.msg: MessageOrchestrator | None = None

        # Résolution des répertoires d'entrée/sortie
        if self.ctx:
            self.in_dir = Path(self.ctx[CTX_DATA_IN])
            self.out_dir = Path(self.ctx[CTX_DATA_OUT])
        else:
            root = Path(get_original_cwd())
            base = (root / self.cfg.data_dir).resolve()
            self.in_dir = (base / self.cfg.in_dir).resolve()
            self.out_dir = (base / self.cfg.out_dir).resolve()

        # S’assurer de l’existence des répertoires
        self.fm.ensure_dir(self.in_dir)
        self.fm.ensure_dir(self.out_dir)

    @classmethod
    def from_cfg_mgr(cls, cfg_mgr: Any, logger_manager: LoggerManager | None = None, ctx: dict[str, str] | None = None) -> FileOrchestrator:
        """Alias legacy; préférer from_config_manager pour homogénéité."""
        cfg = cfg_mgr.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager, ctx=ctx)

    @classmethod
    def from_config_manager(cls, config_manager: Any, logger_manager: LoggerManager | None = None, ctx: dict[str, str] | None = None) -> FileOrchestrator:
        """Constructeur recommandé aligné sur le nommage config_manager."""
        cfg = config_manager.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager, ctx=ctx)

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attache l’orchestrateur de messages pour les émissions localisées."""
        self.msg = msg

    def pick_input_file(self) -> Path | None:
        """
        Sélectionne le fichier d’entrée:
        - Si preferred_filename est défini et existe dans in_dir, il est choisi.
        - Sinon, premier fichier correspondant aux extensions configurées.
        """
        preferred = self.cfg.preferred_filename
        if preferred:
            candidate = (self.in_dir / preferred).resolve()
            if candidate.exists():
                return candidate
        files = self.fm.list_files_by_ext(self.in_dir, self.cfg.extensions)
        return files[0] if files else None

    def process_input(self) -> dict[str, Any]:  # noqa: C901, PLR0912
        """
        Orchestration complète:
        - Émet "file_init", résout/valide les chemins, cherche un fichier entrant.
        - Copie/compresse si configuré, lit le contenu, puis émet "input_processed" avec métadonnées.
        - Retourne un dictionnaire standardisé avec présence, chemins et données.
        """
        # Événement de départ
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

        # Chemins résolus
        if self.msg:
            self.msg.emit(DOMAIN, EV_FILE_PATHS_RESOLVED, **{FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir)})
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                extra_payload = {LOG_EXTRA: {FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir)}}
                logger.info(EV_FILE_PATHS_RESOLVED, extra=extra_payload)

        # Recherche d'un fichier d'entrée
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

        # Fichier trouvé
        if self.msg:
            self.msg.emit(DOMAIN, _evt(INPUT_FOUND), **{FIELD_FILE: str(f)})
        else:
            logger = getattr(self, "log", None)
            if logger is not None:
                logger.info("input_found", extra={LOG_EXTRA: {FIELD_FILE: str(f)}})

        # Persistance/Compression optionnelles
        saved_path: Path | None = None
        compressed_path: Path | None = None
        if self.cfg.save_input_file:
            stamped = self.fm.make_timestamp_name(f)
            saved_path = self.fm.copy_file(f, self.out_dir, rename=stamped)
            if self.cfg.save_input_file_compression and saved_path is not None:
                compressed_path = self.fm.compress_file_gz(saved_path, delete_original=True)

        # Lecture du fichier
        data = self.fm.read_file(f)

        # Événement de fin
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

        # Sortie standardisée
        return {
            KEY_FOUND: True,
            KEY_FILE: str(f),
            KEY_SAVED: (str(saved_path) if saved_path is not None else None),
            KEY_COMPRESSED: (str(compressed_path) if compressed_path is not None else None),
            KEY_DATA: data,
            KEY_META: {FIELD_IN_DIR: str(self.in_dir), FIELD_OUT_DIR: str(self.out_dir), FIELD_EXTENSIONS: self.cfg.extensions},
        }
