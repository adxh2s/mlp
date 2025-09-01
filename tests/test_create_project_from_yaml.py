#!/usr/bin/env python3
"""
Create project directories and files from a YAML specification.

- Reads project_structure.yaml from the current working directory.
- Creates missing directories and files without overwriting existing content.
- Applies permission overrides based on glob-like patterns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("Please install PyYAML: pip install pyyaml")
    sys.exit(1)

STRUCTURE_FILE = "project_structure.yaml"

# Defaults (octal)
DIR_MODE_DEFAULT = 0o755  # rwxr-xr-x
FILE_MODE_DEFAULT = 0o644  # rw-r--r--
FILE_MODE_EXECUTABLE = 0o755  # rwxr-xr-x

EXECUTABLE_EXTS = {".py", ".sh"}
TEXT_EXTS = {".yaml", ".yml", ".md", ".txt", ".json", ".csv", ".ini", ".cfg"}


def log(msg: str) -> None:
    """Print a namespaced log message."""
    print(f"[create_project] {msg}")


def norm_mode(value: Any) -> int:
    """
    Normalize a permission value to an octal int for os.chmod.

    Accepted inputs:
    - int like 755 or 644
    - str like "755", "0644", "0o755"

    Returns:
        int: The normalized octal mode (e.g., 0o755).
    """
    if isinstance(value, int):
        s = str(value)
    elif isinstance(value, str):
        s = value.strip().lower()
        if s.startswith("0o"):
            try:
                return int(s, 8)
            except Exception:
                # Fall-through to digit validation below
                pass
        s = s.lstrip("0") or "0"
    else:
        raise ValueError(f"Invalid mode type: {type(value)} ({value})")

    if not s.isdigit():
        raise ValueError(f"Invalid mode digits: {value}")

    acc = 0
    for ch in s:
        d = ord(ch) - ord("0")
        if d < 0 or d > 7:
            raise ValueError(f"Invalid octal digit in mode: {value}")
        acc = (acc << 3) | d
    return acc


def make_dir(path: Path, mode: int) -> None:
    """
    Ensure a directory exists and apply permissions.

    Creates the directory tree if missing. If the directory already exists,
    it is left intact; only permissions are applied.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, mode)
        log(f"DIR OK {path} mode={oct(mode)}")
    except Exception as exc:
        log(f"DIR ERR {path} -> {exc}")


def make_file(path: Path, mode: int) -> None:
    """
    Ensure a file exists and apply permissions.

    If the file does not exist, create an empty file. If it exists,
    do not overwrite or update its content or mtime intentionally;
    only permissions are applied.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch(exist_ok=True)
            log(f"FILE NEW {path}")
        else:
            log(f"FILE EXIST {path}")
        os.chmod(path, mode)
        log(f"FILE MODE {path} -> {oct(mode)}")
    except Exception as exc:
        log(f"FILE ERR {path} -> {exc}")


def is_file_name(name: str) -> bool:
    """
    Heuristically decide if a mapping key denotes a file (contains a dot).

    Args:
        name: Mapping key from the YAML tree.

    Returns:
        True if the key likely denotes a file name, False for a directory.
    """
    has_dot = "." in name
    log(f"is_file_name? name='{name}' -> {has_dot}")
    return has_dot


def load_permissions_overrides(struct: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Extract and normalize file/dir permission overrides from the structure.

    The structure should include:
      permissions:
        files:
          "<glob>": <mode>
        dirs:
          "<glob>": <mode>

    Returns:
        Dict with "files" and "dirs" mapping glob patterns to octal modes.
    """
    perms = struct.get("permissions", {}) or {}
    file_map_raw = perms.get("files") or {}
    dir_map_raw = perms.get("dirs") or {}

    def norm_map(mapping: Dict[str, Any]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for pattern, value in mapping.items():
            try:
                out[pattern] = norm_mode(value)
            except Exception as exc:
                log(
                    "PERM WARN invalid mode for pattern "
                    f"'{pattern}': {value} ({exc}) -> skipped"
                )
        return out

    file_map = norm_map(file_map_raw)
    dir_map = norm_map(dir_map_raw)
    log(
        "PERMISSIONS overrides loaded (octal): "
        f"files={{{k: oct(v) for k, v in file_map.items()}}}, "
        f"dirs={{{k: oct(v) for k, v in dir_map.items()}}}"
    )
    return {"files": file_map, "dirs": dir_map}


def pick_mode_for_dir(path: Path, overrides: Dict[str, int]) -> int:
    """
    Select a permission mode for a directory, honoring glob overrides.

    Returns:
        int: The selected permission mode.
    """
    for pattern, mode in overrides.items():
        if path.match(pattern):
            log(f"DIR MODE override match: {path} matches '{pattern}' -> {oct(mode)}")
            return mode
    log(f"DIR MODE default for {path} -> {oct(DIR_MODE_DEFAULT)}")
    return DIR_MODE_DEFAULT


def pick_mode_for_file(path: Path, overrides: Dict[str, int]) -> int:
    """
    Select a permission mode for a file, honoring glob overrides and extensions.

    Precedence:
    1) Explicit glob override
    2) Executable by extension (py/sh)
    3) Text default (yaml/yml/md/txt/json/csv/ini/cfg)
    4) Fallback default
    """
    for pattern, mode in overrides.items():
        if path.match(pattern):
            log(
                f"FILE MODE override match: {path} matches '{pattern}' -> {oct(mode)}"
            )
            return mode

    ext = path.suffix.lower()
    if ext in EXECUTABLE_EXTS:
        log(f"FILE MODE executable by extension: {path} -> {oct(FILE_MODE_EXECUTABLE)}")
        return FILE_MODE_EXECUTABLE

    if ext in TEXT_EXTS:
        log(f"FILE MODE text default: {path} -> {oct(FILE_MODE_DEFAULT)}")
        return FILE_MODE_DEFAULT

    log(f"FILE MODE fallback default: {path} -> {oct(FILE_MODE_DEFAULT)}")
    return FILE_MODE_DEFAULT


def ensure_children(
    base: Path,
    children: Dict[str, Any],
    created: Dict[str, list],
    file_perm_over: Dict[str, int],
    dir_perm_over: Dict[str, int],
) -> None:
    """
    Recursively create directories and files under a base directory.

    Args:
        base: Current base directory path.
        children: Mapping of names to nested nodes.
        created: Accumulator of created dirs/files as string paths.
        file_perm_over: File permission override mapping.
        dir_perm_over: Directory permission override mapping.
    """
    if not isinstance(children, dict):
        log(f"CHILDREN WARN at {base}: expected dict, got {type(children)}")
        return

    for name, node in children.items():
        log(
            "CHILD PROC "
            f"base={base} name={name} node_type={type(node)} "
            f"node={node if isinstance(node, (str, int, float)) else '[complex]'}"
        )
        target = base / name

        if is_file_name(name):
            mode = pick_mode_for_file(target, file_perm_over)
            make_file(target, mode)
            created["files"].append(str(target))
            continue

        # Directory branch
        make_dir(target, pick_mode_for_dir(target, dir_perm_over))
        created["dirs"].append(str(target))

        if isinstance(node, dict):
            nested = node.get("children")
            if isinstance(nested, dict):
                ensure_children(target, nested, created, file_perm_over, dir_perm_over)
        else:
            # Mixed mapping under directory: nested explicit keys
            if isinstance(node, dict):
                for sub_name, sub_node in node.items():
                    if sub_name == "children":
                        continue
                    sub_target = target / sub_name
                    log(
                        "DIR MIXED "
                        f"base={target} sub_name={sub_name} sub_type={type(sub_node)}"
                    )
                    if is_file_name(sub_name):
                        mode = pick_mode_for_file(sub_target, file_perm_over)
                        make_file(sub_target, mode)
                        created["files"].append(str(sub_target))
                    elif isinstance(sub_node, dict):
                        make_dir(sub_target, pick_mode_for_dir(sub_target, dir_perm_over))
                        created["dirs"].append(str(sub_target))
                        maybe_children = sub_node.get("children")
                        if isinstance(maybe_children, dict):
                            ensure_children(
                                sub_target,
                                maybe_children,
                                created,
                                file_perm_over,
                                dir_perm_over,
                            )


def process_root_structure(struct: Dict[str, Any], root: Path) -> Dict[str, list]:
    """
    Process the loaded YAML structure for a project at the given root path.

    Returns:
        Dict[str, list]: Paths of created directories and files.
    """
    created: Dict[str, list] = {"dirs": [], "files": []}

    project = struct.get("project", {})
    log(f"ROOT project keys: {list(project.keys())}")

    perms = load_permissions_overrides(struct)
    file_perm_over = perms.get("files", {})
    dir_perm_over = perms.get("dirs", {})

    directories = project.get("directories", {})
    if not isinstance(directories, dict):
        log(f"DIRECTORIES WARN: expected dict, got {type(directories)}")
    else:
        for dirname, meta in directories.items():
            dir_path = root / dirname
            log(f"TOP DIR {dir_path} meta_type={type(meta)}")
            make_dir(dir_path, pick_mode_for_dir(dir_path, dir_perm_over))
            created["dirs"].append(str(dir_path))

            if isinstance(meta, dict):
                for maybe_name, maybe_node in meta.items():
                    if maybe_name == "children":
                        continue
                    if is_file_name(maybe_name):
                        fpath = dir_path / maybe_name
                        mode = pick_mode_for_file(fpath, file_perm_over)
                        make_file(fpath, mode)
                        created["files"].append(str(fpath))

                children = meta.get("children")
                if isinstance(children, dict):
                    ensure_children(
                        dir_path, children, created, file_perm_over, dir_perm_over
                    )

    files = project.get("files", {})
    if not isinstance(files, dict):
        log(f"FILES WARN: expected dict, got {type(files)}")
    else:
        for fname, _desc in files.items():
            fpath = root / fname
            log(f"TOP FILE {fpath}")
            mode = pick_mode_for_file(fpath, file_perm_over)
            make_file(fpath, mode)
            created["files"].append(str(fpath))

    return created


def main() -> None:
    """Entry-point: load YAML, create filesystem layout, and print a summary."""
    cwd = Path.cwd()
    struct_path = cwd / STRUCTURE_FILE
    log(f"START in {cwd}")
    log(f"STRUCT file path: {struct_path}")

    if not struct_path.exists():
        print(f"Structure file not found at {struct_path}.")
        print("Place project_structure.yaml in the current working directory and rerun.")
        sys.exit(1)

    try:
        with struct_path.open("r", encoding="utf-8") as f:
            struct = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"YAML parsing error: {exc}")
        sys.exit(1)

    if not isinstance(struct, dict) or "project" not in struct:
        print("Invalid structure file: missing top-level 'project' mapping.")
        log(
            "STRUCT root type: "
            f"{type(struct)}, keys: {list(struct.keys()) if isinstance(struct, dict) else None}"
        )
        sys.exit(1)

    log(f"STRUCT loaded. Top-level keys: {list(struct.keys())}")
    created = process_root_structure(struct, cwd)

    print("Creation complete.")
    print(f"- Directories created: {len(created['dirs'])}")
    print(f"- Files created: {len(created['files'])}")
    if created["dirs"]:
        print("Directories:")
        for d in created["dirs"]:
            print(f" - {d}")
    if created["files"]:
        print("Files:")
        for f in created["files"]:
            print(f" - {f}")


if __name__ == "__main__":
    main()
