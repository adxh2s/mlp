from __future__ import annotations

"""
File orchestrator: locate, optionally persist, and load input files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config.schemas import FileConfig as FileConfigModel
from src.instrumentation.logger_mixin import LoggerMixin
from src.orchestrators.file import FileManager  # keep existing import
from src.orchestrators.messages import MessageOrchestrator
from src.instrumentation.messages_taxonomy import (
    FILE_INIT,
    NO_INPUT_FILE,
    INPUT_FOUND,
    INPUT_PROCESSED,
)
from src.instrumentation.logger_manager import LoggerManager

# Constants
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
    """Configuration for FileOrchestrator."""

    enabled: bool = True
    data_dir: str = "./data"
    in_dir: str = "in"
    out_dir: str = "out"
    extensions: list[str] = field(
        default_factory=lambda: [".csv", ".xlsx", ".json"]
    )
    save_input_file: bool = True
    save_input_file_compression: bool = False


class FileOrchestrator(LoggerMixin):
    """
    Locate, optionally persist, and load input files for downstream tasks.
    """

    def __init__(
        self,
        cfg: FileConfigModel | FileConfig,
        logger_manager: Optional[LoggerManager] = None,
    ) -> None:
        """
        Initialize the orchestrator and ensure input/output directories exist.

        Args:
            cfg: File configuration (Pydantic or dataclass).
            logger_manager: Optional LoggerManager for structured logging.
        """
        # Normalize config to dataclass for uniformity
        if hasattr(cfg, "model_dump"):
            d = cfg.model_dump()
            self.cfg = FileConfig(**d)
        elif isinstance(cfg, FileConfig):
            self.cfg = cfg
        else:
            self.cfg = FileConfig(**dict(cfg))

        self.fm = FileManager()

        self.LOGGER_NAME = LOGGER_NAME
        if logger_manager is not None:
            self._init_logger(logger_manager)
        else:
            import logging

            self.log = logging.getLogger(LOGGER_NAME)

        self.msg: Optional[MessageOrchestrator] = None  # may be injected

        data_root = Path(self.cfg.data_dir).expanduser().resolve()
        self.in_dir = (
            Path(self.cfg.in_dir).resolve()
            if Path(self.cfg.in_dir).is_absolute()
            else (data_root / self.cfg.in_dir).resolve()
        )
        self.out_dir = (
            Path(self.cfg.out_dir).resolve()
            if Path(self.cfg.out_dir).is_absolute()
            else (data_root / self.cfg.out_dir).resolve()
        )

        self.fm.ensure_dir(self.in_dir)
        self.fm.ensure_dir(self.out_dir)

        if self.msg:
            self.msg.emit(
                DOMAIN,
                FILE_INIT,
                in_dir=str(self.in_dir),
                out_dir=str(self.out_dir),
            )
        else:
            self.log.info(
                "file_orchestrator_init",
                extra={
                    "extra_fields": {
                        "in_dir": str(self.in_dir),
                        "out_dir": str(self.out_dir),
                    }
                },
            )

    @classmethod
    def from_cfg_mgr(
        cls, cfg_mgr, logger_manager: Optional[LoggerManager] = None
    ) -> "FileOrchestrator":
        """Build a FileOrchestrator from a ConfigManager/AppConfig tree."""
        cfg = cfg_mgr.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager)

    def attach_messages(self, msg: MessageOrchestrator) -> None:
        """Attach a MessageOrchestrator for localized emissions."""
        self.msg = msg

    def pick_input_file(self) -> Path | None:
        """Return a first matching input file by configured extensions, or None."""
        files = self.fm.list_files_by_ext(self.in_dir, self.cfg.extensions)
        return files if files else None

    def process_input(self) -> dict[str, Any]:
        """
        Process input pipeline: find, optionally persist, and load data.

        Returns:
            dict[str, Any]: keys "found", "file", "saved_copy",
            "saved_copy_compressed", "data", and "meta".
        """
        f = self.pick_input_file()
        if f is None:
            if self.msg:
                self.msg.emit(DOMAIN, NO_INPUT_FILE)
            else:
                self.log.info("no_input_file")
            return {
                KEY_FOUND: False,
                KEY_FILE: None,
                KEY_SAVED: None,
                KEY_COMPRESSED: None,
                KEY_DATA: None,
                KEY_META: {
                    "in_dir": str(self.in_dir),
                    "out_dir": str(self.out_dir),
                    "extensions": self.cfg.extensions,
                },
            }

        if self.msg:
            self.msg.emit(DOMAIN, INPUT_FOUND, file=str(f))
        else:
            self.log.info(
                "input_found", extra={"extra_fields": {"file": str(f)}}
            )

        saved_path: Path | None = None
        compressed_path: Path | None = None

        if self.cfg.save_input_file:
            stamped = self.fm.make_timestamp_name(f)
            saved_path = self.fm.copy_file(f, self.out_dir, rename=stamped)
            if self.cfg.save_input_file_compression and saved_path:
                compressed_path = self.fm.compress_file_gz(
                    saved_path, delete_original=True
                )

        data = self.fm.read_file(f)

        if self.msg:
            self.msg.emit(
                DOMAIN,
                INPUT_PROCESSED,
                file=str(f),
                saved=str(saved_path) if saved_path else None,
                compressed=str(compressed_path) if compressed_path else None,
            )
        else:
            self.log.info(
                "input_processed",
                extra={
                    "extra_fields": {
                        "file": str(f),
                        "saved": str(saved_path) if saved_path else None,
                        "compressed": (
                            str(compressed_path) if compressed_path else None
                        ),
                    }
                },
            )

        return {
            KEY_FOUND: True,
            KEY_FILE: str(f),
            KEY_SAVED: str(saved_path) if saved_path else None,
            KEY_COMPRESSED: str(compressed_path) if compressed_path else None,
            KEY_DATA: data,
            KEY_META: {
                "in_dir": str(self.in_dir),
                "out_dir": str(self.out_dir),
                "extensions": self.cfg.extensions,
            },
        }
