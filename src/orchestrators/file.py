from __future__ import annotations

"""
File orchestrator: locate, optionally persist, and load input files.

- Uses ctx["data_in"]/ctx["data_out"] when provided (from AppOrchestrator).
- Falls back to Hydra-anchored resolution via get_original_cwd() otherwise.
- Emits localized events for transparent, structured observability.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hydra.utils import get_original_cwd

from src.config.schemas import FileConfig as FileConfigModel
from src.instrumentation.file_manager import FileManager
from src.instrumentation.logger_manager import LoggerManager
from src.instrumentation.logger_mixin import LoggerMixin
from src.instrumentation.messages_taxonomy import FILE_INIT, INPUT_FOUND, INPUT_PROCESSED, NO_INPUT_FILE
from src.orchestrators.messages import MessageOrchestrator

LOGGER_NAME = "mlp.orchestrators.file"
DOMAIN = "file"

KEY_FOUND = "found"
KEY_FILE = "file"
KEY_SAVED = "saved_copy"
KEY_COMPRESSED = "saved_copy_compressed"
KEY_DATA = "data"
KEY_META = "meta"


@dataclass(slots=True)
class FileConfig:
    enabled: bool = True
    data_dir: str = "data"
    in_dir: str = "in"
    out_dir: str = "out"
    extensions: list[str] = field(default_factory=lambda: [".csv", ".xlsx", ".json"])
    save_input_file: bool = True
    save_input_file_compression: bool = False
    # preferred_filename: str | None = None


class FileOrchestrator(LoggerMixin):
    """Locate, optionally persist, and load input files for downstream tasks."""

    def __init__(self, cfg: FileConfigModel | FileConfig, logger_manager: Optional[LoggerManager] = None, ctx: Optional[dict[str, str]] = None) -> None:
        # Normalize config
        if hasattr(cfg, "model_dump"):
            d = cfg.model_dump()
            self.cfg = FileConfig(**d)
        elif isinstance(cfg, FileConfig):
            self.cfg = cfg
        else:
            self.cfg = FileConfig(**dict(cfg))

        self.ctx = ctx or {}
        self.fm = FileManager()
        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager is not None:
            self._init_logger(logger_manager)
        else:
            import logging

            self.log = logging.getLogger(LOGGER_NAME)

        self.msg: Optional[MessageOrchestrator] = None

        # Pre-resolve dirs (prefer ctx)
        if self.ctx:
            self.in_dir = Path(self.ctx["data_in"])
            self.out_dir = Path(self.ctx["data_out"])
        else:
            data_root = Path(self.cfg.data_dir).expanduser().resolve()
            self.in_dir = (Path(self.cfg.in_dir).resolve() if Path(self.cfg.in_dir).is_absolute() else (data_root / self.cfg.in_dir).resolve())
            self.out_dir = (Path(self.cfg.out_dir).resolve() if Path(self.cfg.out_dir).is_absolute() else (data_root / self.cfg.out_dir).resolve())
        self.fm.ensure_dir(self.in_dir)
        self.fm.ensure_dir(self.out_dir)

    @classmethod
    def from_cfg_mgr(cls, cfg_mgr, logger_manager: Optional[LoggerManager] = None, ctx: Optional[dict[str, str]] = None) -> "FileOrchestrator":
        cfg = cfg_mgr.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager, ctx=ctx)

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        self.msg = msg

    def pick_input_file(self) -> Path | None:
        preferred = getattr(self.cfg, "preferred_filename", None)
        if preferred:
            candidate = (self.in_dir / preferred).resolve()
            if candidate.exists():
                return candidate
        files = self.fm.list_files_by_ext(self.in_dir, self.cfg.extensions)
        return files[0] if files else None

    def process_input(self) -> dict[str, Any]:
        if self.msg:
            self.msg.emit(DOMAIN, FILE_INIT, data_dir=self.cfg.data_dir, in_dir=self.cfg.in_dir, out_dir=self.cfg.out_dir)
        else:
            self.log.info("file_init", extra={"extra_fields": {"data_dir": self.cfg.data_dir, "in_dir": self.cfg.in_dir, "out_dir": self.cfg.out_dir}})

        if not self.ctx:
            root = Path(get_original_cwd())
            base = (root / self.cfg.data_dir).resolve()
            self.in_dir = (base / self.cfg.in_dir).resolve()
            self.out_dir = (base / self.cfg.out_dir).resolve()
            self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.msg:
            self.msg.emit(DOMAIN, "file_paths_resolved", in_dir=str(self.in_dir), out_dir=str(self.out_dir))
        else:
            self.log.info("file_paths_resolved", extra={"extra_fields": {"in_dir": str(self.in_dir), "out_dir": str(self.out_dir)}})

        if self.msg:
            self.msg.emit(DOMAIN, "file_pick_start", exts=self.cfg.extensions)
        else:
            self.log.info("file_pick_start", extra={"extra_fields": {"extensions": self.cfg.extensions}})

        f = self.pick_input_file()

        if f is None:
            if self.msg:
                self.msg.emit(DOMAIN, NO_INPUT_FILE, in_dir=str(self.in_dir), exts=self.cfg.extensions)
            else:
                self.log.info("no_input_file", extra={"extra_fields": {"in_dir": str(self.in_dir), "extensions": self.cfg.extensions}})
            return {
                KEY_FOUND: False,
                KEY_FILE: None,
                KEY_SAVED: None,
                KEY_COMPRESSED: None,
                KEY_DATA: None,
                KEY_META: {"in_dir": str(self.in_dir), "out_dir": str(self.out_dir), "extensions": self.cfg.extensions},
            }

        if self.msg:
            self.msg.emit(DOMAIN, INPUT_FOUND, file=str(f))
        else:
            self.log.info("input_found", extra={"extra_fields": {"file": str(f)}})

        saved_path: Path | None = None
        compressed_path: Path | None = None

        if self.cfg.save_input_file:
            stamped = self.fm.make_timestamp_name(f)
            saved_path = self.fm.copy_file(f, self.out_dir, rename=stamped)
            if self.cfg.save_input_file_compression and saved_path:
                compressed_path = self.fm.compress_file_gz(saved_path, delete_original=True)

        data = self.fm.read_file(f)

        if self.msg:
            self.msg.emit(DOMAIN, INPUT_PROCESSED, file=str(f), saved=str(saved_path) if saved_path else None, compressed=str(compressed_path) if compressed_path else None)
        else:
            self.log.info("input_processed", extra={"extra_fields": {"file": str(f), "saved": str(saved_path) if saved_path else None, "compressed": str(compressed_path) if compressed_path else None}})

        return {
            KEY_FOUND: True,
            KEY_FILE: str(f),
            KEY_SAVED: str(saved_path) if saved_path else None,
            KEY_COMPRESSED: str(compressed_path) if compressed_path else None,
            KEY_DATA: data,
            KEY_META: {"in_dir": str(self.in_dir), "out_dir": str(self.out_dir), "extensions": self.cfg.extensions},
        }
