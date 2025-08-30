from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def index_path(project_root: Path) -> Path:
    return project_root / "runs_index.json"


def load_runs_index(project_root: Path) -> List[Dict]:
    p = index_path(project_root)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def append_run(project_root: Path, run: Dict) -> None:
    arr = load_runs_index(project_root)
    arr.append(run)
    index_path(project_root).write_text(json.dumps(arr, indent=2, ensure_ascii=False), encoding="utf-8")
