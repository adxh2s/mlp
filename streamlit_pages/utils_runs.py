from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

"""Utilitaires de suivi des exécutions (runs) pour l'UI Streamlit."""


def new_run_id() -> str:
    """Retourne un identifiant horodaté YYYYmmdd_HHMMSS."""
    return time.strftime("%Y%m%d_%H%M%S")


def index_path(project_root: Path) -> Path:
    """Chemin du fichier d'index des runs."""
    return project_root / "runs_index.json"


def load_runs_index(project_root: Path) -> list[dict[str, Any]]:
    """Charge l'index des runs s'il existe, sinon une liste vide."""
    p = index_path(project_root)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def append_run(project_root: Path, run: dict[str, Any]) -> None:
    """Ajoute un run dans l'index JSON."""
    arr = load_runs_index(project_root)
    arr.append(run)
    index_path(project_root).write_text(json.dumps(arr, indent=2, ensure_ascii=False), encoding="utf-8")
