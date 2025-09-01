from __future__ import annotations

# Standard library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Third-party
# (none)
# Local
from src.config.schemas import FileConfig as FileConfigModel
from src.instrumentation.logger_mixin import LoggerMixin
from src.orchestrators.file import FileManager


@dataclass(slots=True)
class FileConfig:
    """Configuration for FileOrchestrator.

    Attributes:
        enabled: Whether the file orchestrator is active.
        data_dir: Root data directory.
        in_dir: Input directory (absolute or relative to data_dir).
        out_dir: Output directory (absolute or relative to data_dir).
        extensions: Allowed extensions (e.g., [".csv", ".xlsx", ".json"]).
        save_input_file: If True, copy input to out_dir before processing.
        save_input_file_compression: If True, gzip the saved copy.
    """

    enabled: bool = True
    data_dir: str = "./data"
    in_dir: str = "in"
    out_dir: str = "out"
    extensions: list[str] = field(default_factory=lambda: [".csv", ".xlsx", ".json"])
    save_input_file: bool = True
    save_input_file_compression: bool = False


class FileOrchestrator(LoggerMixin):
    """Locate, optionally persist, and load input files for downstream tasks.

    Resolves in/out directories, lists by extensions, picks a candidate,
    optionally copies and gzips the input, reads data, and returns metadata.
    """

    # Class constants
    LOGGER_NAME = "mlp.orchestrators.file"
    KEY_FOUND = "found"
    KEY_FILE = "file"
    KEY_SAVED = "saved_copy"
    KEY_COMPRESSED = "saved_copy_compressed"
    KEY_DATA = "data"
    KEY_META = "meta"

    def __init__(self, cfg: FileConfigModel | FileConfig, logger_manager=None) -> None:
        """Initialize the orchestrator and ensure input/output directories exist.

        Args:
            cfg: File configuration (Pydantic FileConfigModel or dataclass FileConfig).
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
        self.lm = logger_manager

        if self.lm is not None:
            self._init_logger(self.lm)
        else:
            import logging
            self.log = logging.getLogger(self.LOGGER_NAME)

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

        self.log.info(
            "file_orchestrator_init",
            extra={"extra_fields": {"in_dir": str(self.in_dir), "out_dir": str(self.out_dir)}},
        )

    @classmethod
    def from_cfg_mgr(cls, cfg_mgr, logger_manager=None) -> "FileOrchestrator":
        """Build a FileOrchestrator from a ConfigManager/AppConfig tree."""
        cfg = cfg_mgr.model.orchestrators.file
        return cls(cfg, logger_manager=logger_manager)

    def pick_input_file(self) -> Path | None:
        """Return a first matching input file by configured extensions, or None."""
        files = self.fm.list_files_by_ext(self.in_dir, self.cfg.extensions)
        return files if files else None

    def process_input(self) -> dict[str, Any]:
        """Process input pipeline: find, optionally persist, and load data.

        Returns:
            dict[str, Any]: Keys "found", "file", "saved_copy", "saved_copy_compressed",
            "data", and "meta" with directory and extension details.
        """
        f = self.pick_input_file()
        if f is None:
            self.log.info("no_input_file")
            return {
                self.KEY_FOUND: False,
                self.KEY_FILE: None,
                self.KEY_SAVED: None,
                self.KEY_COMPRESSED: None,
                self.KEY_DATA: None,
                self.KEY_META: {
                    "in_dir": str(self.in_dir),
                    "out_dir": str(self.out_dir),
                    "extensions": self.cfg.extensions,
                },
            }

        self.log.info("input_found", extra={"extra_fields": {"file": str(f)}})

        saved_path: Path | None = None
        compressed_path: Path | None = None

        if self.cfg.save_input_file:
            stamped = self.fm.make_timestamp_name(f)
            saved_path = self.fm.copy_file(f, self.out_dir, rename=stamped)
            if self.cfg.save_input_file_compression:
                compressed_path = self.fm.compress_file_gz(saved_path, delete_original=True)

        data = self.fm.read_file(f)

        self.log.info(
            "input_processed",
            extra={"extra_fields": {
                "file": str(f),
                "saved": str(saved_path) if saved_path else None,
                "compressed": str(compressed_path) if compressed_path else None,
            }},
        )

        return {
            self.KEY_FOUND: True,
            self.KEY_FILE: str(f),
            self.KEY_SAVED: str(saved_path) if saved_path else None,
            self.KEY_COMPRESSED: str(compressed_path) if compressed_path else None,
            self.KEY_DATA: data,
            self.KEY_META: {
                "in_dir": str(self.in_dir),
                "out_dir": str(self.out_dir),
                "extensions": self.cfg.extensions,
            },
        }
