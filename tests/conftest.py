from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import logging
from omegaconf import OmegaConf
from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager

import pandas as pd
import pytest


def pytest_configure(config):
    # Config minimale compatible avec AppConfig
    raw = {
        "project": {"name": "tests", "random_state": 42, "output_dir": str(Path("tmp_pytest_outputs").resolve())},
        "orchestrators": {
            "eda": {"enabled": False},
            "pipelines": {"enabled": False, "cv": {"cv_folds": 2, "scoring": ["f1"]}, "pipelines": []},
            "report": {"enabled": False, "formats": ["md"]},
        },
        "logger": {"backend": "stdlib", "level": "INFO"},  # file_path sera injecté par ConfigManager
    }
    cfg = OmegaConf.create(raw)
    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()

    # LoggerSettings -> file_path absolu <racine>/logs/app.log si absent
    ls = cfg_mgr.build_logger_settings()
    # Créer dossier parent
    if ls.file_path:
        Path(ls.file_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    # Construire et configurer
    lm = build_logger_manager(ls)
    lm.configure()

    # Log d’amorçage
    import logging
    logging.getLogger(__name__).info("pytest logging configured", extra={"extra_fields": {"log_file": ls.file_path}})



@pytest.fixture(scope="session")
def tmp_outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped outputs directory for artifacts."""
    p = tmp_path_factory.mktemp("outputs")
    return p


@pytest.fixture()
def demo_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Small demo dataset (10 rows) for fast tests."""
    import numpy as np

    X = pd.DataFrame(
        {
            "x1": np.arange(10),
            "x2": np.linspace(0.0, 1.0, 10),
            "x3": [0, 1] * 5,
        }
    )
    vals = [0, 1] * 5 
    y = pd.Series(vals, name="target")
    return X, y


@pytest.fixture()
def jinja_templates_dir(tmp_path: Path) -> Path:
    """Create a small Jinja templates dir for renderer tests."""
    base = tmp_path / "templates"
    base.mkdir(parents=True, exist_ok=True)
    (base / "base.html.jinja").write_text("<html><body>{% block body %}{% endblock %}</body></html>", encoding="utf-8")
    (base / "report.html.jinja").write_text("{% extends 'base.html.jinja' %}{% block body %}<h1>{{ project_name }}</h1>{% endblock %}", encoding="utf-8")
    (base / "report.md.jinja").write_text("# {{ project_name }}", encoding="utf-8")
    return base
