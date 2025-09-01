from __future__ import annotations

# Standard library
import gzip
import os
import shutil
from collections.abc import Iterable  # Ruff UP035
from datetime import datetime
from pathlib import Path
from typing import Any

# Third-party
import pandas as pd


class FileManager:
    """File and directory utilities with pragmatic defaults.

    Provides helpers for:
    - Existence and type checks.
    - Permission checks (read/write/execute).
    - Directory creation (idempotent).
    - Listing files by extension.
    - Copying files with optional rename.
    - Timestamped naming.
    - Gzip compression for single files.
    - Read/write tabular files (csv, xlsx, json).
    """

    # Class constants
    TS_FMT = "%Y%m%d_%H%M%S"
    SUPPORTED_READ = {".csv", ".xlsx", ".xls", ".json"}
    DEFAULT_JSON_ORIENT = "records"

    # Error messages
    ERR_UNSUPPORTED_EXT = "Unsupported extension: {}"

    def check_path_exists(self, path: str | Path) -> bool:
        """Return True if path exists (file or directory), else False."""
        return Path(path).exists()

    def is_file(self, path: str | Path) -> bool:
        """Return True if path points to a regular file."""
        return Path(path).is_file()

    def is_dir(self, path: str | Path) -> bool:
        """Return True if path points to a directory."""
        return Path(path).is_dir()

    def has_perm_read(self, path: str | Path) -> bool:
        """Return True if current process can read path."""
        return os.access(str(path), os.R_OK)

    def has_perm_write(self, path: str | Path) -> bool:
        """Return True if current process can write to path."""
        return os.access(str(path), os.W_OK)

    def has_perm_exec(self, path: str | Path) -> bool:
        """Return True if current process can execute path."""
        return os.access(str(path), os.X_OK)

    def ensure_dir(self, path: str | Path) -> Path:
        """Create directory path with parents, no error if exists."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_files_by_ext(self, dir_path: str | Path, exts: Iterable[str]) -> list[Path]:
        """List files in dir_path whose suffix is in exts (case-insensitive, non-recursive)."""
        base = Path(dir_path)
        wanted = {e.lower() for e in exts}
        out: list[Path] = []
        if base.is_dir():
            for p in base.iterdir():
                if p.is_file() and p.suffix.lower() in wanted:
                    out.append(p)
        return sorted(out)

    def make_timestamp_name(self, src: str | Path) -> str:
        """Return a timestamped filename YYYYMMDD_HHMMSS_<basename>."""
        p = Path(src)
        ts = datetime.now().strftime(self.TS_FMT)
        return f"{ts}_{p.name}"

    def copy_file(self, src: str | Path, dst_dir: str | Path, rename: str | None = None) -> Path:
        """Copy file to dst_dir, optionally renaming the target (preserves metadata)."""
        self.ensure_dir(dst_dir)
        src_p = Path(src)
        dst_name = rename if rename else src_p.name
        dst_p = Path(dst_dir) / dst_name
        return Path(shutil.copy2(src_p, dst_p))

    def compress_file_gz(self, path: str | Path, delete_original: bool = False) -> Path:
        """Compress single file to gzip (path.ext.gz); optionally delete original."""
        src = Path(path)
        gz_path = src.with_suffix(src.suffix + ".gz")
        with open(src, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        if delete_original:
            src.unlink(missing_ok=True)
        return gz_path

    def read_file(self, path: str | Path) -> Any:
        """Read csv/xlsx/json into a pandas object."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(p)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(p)
        if ext == ".json":
            return pd.read_json(p)
        raise ValueError(self.ERR_UNSUPPORTED_EXT.format(ext))

    def write_file(self, data: Any, path: str | Path) -> None:
        """Write pandas/tabular data to csv/xlsx/json by extension; fallback to text/bytes."""
        p = Path(path)
        self.ensure_dir(p.parent)
        ext = p.suffix.lower()
        if isinstance(data, pd.DataFrame):
            if ext == ".csv":
                data.to_csv(p, index=False)
                return
            if ext in {".xlsx", ".xls"}:
                data.to_excel(p, index=False)
                return
            if ext == ".json":
                data.to_json(p, orient=self.DEFAULT_JSON_ORIENT)
                return
            raise ValueError(self.ERR_UNSUPPORTED_EXT.format(ext))
        mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
        with open(p, mode, encoding=None if mode == "wb" else "utf-8") as f:
            if mode == "wb":
                f.write(data)  # type: ignore[arg-type]
            else:
                f.write(str(data))
