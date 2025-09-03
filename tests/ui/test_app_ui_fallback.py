from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from omegaconf import OmegaConf

import src.ui.app as app_mod
from src.ui.app import MLPStreamlitApp
from src.ui.constants import KEY_APP_TITLE

"""
Tests for UI fallback when no .mo is present.

This test ensures that the class-based Streamlit app:
- Boots with a minimal config (no compiled translations),
- Exposes a public get_ui_text() API,
- Returns default fallback strings for UI texts when no .mo exists.

Conventions:
- No imports inside functions (Ruff PLC0415).
- Avoid bare `assert` (Ruff S101): use explicit checks with AssertionError.
- Type annotations for pytest.MonkeyPatch and local variables (Pylance).
"""


def _write_min_cfg(tmp_path: Path) -> Path:
    """
    Write a minimal valid config file into tmp_path/conf/config.yaml.

    The config keeps only the essentials and includes the messages
    orchestrator with the streamlit_app domain but without any .mo files.
    """
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(
        {
            "project": {"name": "demo", "random_state": 42, "output_dir": "outputs"},
            "orchestrators": {
                "config": {"enabled": True},
                "messages": {
                    "enabled": True,
                    "locale": "fr",
                    "locales_dir": "i18n/locales",
                    "domains": ["streamlit_app"],
                },
            },
            "logger": {
                "backend": "stdlib",
                "app_name": "mlp",
                "level": "INFO",
                "json_mode": False,
            },
        }
    )
    path = conf_dir / "config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    return path


def test_ui_fallback_without_mo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When no .mo exists for streamlit_app, the UI must fall back to defaults.
    """
    # Arrange
    conf_path = _write_min_cfg(tmp_path)
    # Ensure app points to the temp config path
    monkeypatch.setattr(app_mod, "CONF_PATH", conf_path, raising=False)

    # Clear Streamlit session state for test isolation
    st.session_state.clear()

    # Act
    app = MLPStreamlitApp()
    app.bootstrap()
    title: str = app.get_ui_text(KEY_APP_TITLE)

    # Assert (avoid bare assert due to Ruff S101)
    expected = "mlp — Console (Programmatic Routing)"
    if title != expected:
        raise AssertionError(
            f'Expected default fallback title "{expected}", got "{title}"'
        )
