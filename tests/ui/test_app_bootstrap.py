from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

import pytest
import streamlit as st
from omegaconf import OmegaConf

import src.ui.app as app_mod
from src.ui.app import MLPStreamlitApp
from src.ui.constants import KEY_APP_TITLE


def _write_cfg(tmp_path: Path) -> Path:
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
                    "domains": ["general", "config", "streamlit_app"],
                },
            },
            "logger": {"backend": "stdlib", "app_name": "mlp", "level": "INFO", "json_mode": False},
        }
    )
    (conf_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    return conf_dir / "config.yaml"


def _write_streamlit_po(tmp_path: Path) -> None:
    lcd = tmp_path / "i18n" / "locales" / "fr" / "LC_MESSAGES"
    lcd.mkdir(parents=True, exist_ok=True)
    (lcd / "streamlit_app.po").write_text(
        'msgid ""\n'
        'msgstr ""\n'
        '"Project-Id-Version: mlp 1.0\\n"\n'
        '"Language: fr\\n"\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        "\n"
        'msgid "app_title"\n'
        'msgstr "Titre UI Test"\n',
        encoding="utf-8",
    )
    # compile mo if available
    
    if which("msgfmt") is not None:
        mo_path = lcd / "streamlit_app.mo"
        po_path = lcd / "streamlit_app.po"
        msgfmt_path = which("msgfmt")
        # Validate that the files exist and are within the expected directory
        if po_path.exists() and po_path.parent == lcd and mo_path.parent == lcd and msgfmt_path is not None:
            # Validate that msgfmt_path is a safe executable and arguments are safe paths
            allowed_cmd = Path(msgfmt_path).name == "msgfmt"
            allowed_args = all(
                not arg.startswith(";") and not arg.startswith("|")
                for arg in [str(mo_path), str(po_path)]
            )
            if allowed_cmd and allowed_args:
                # Additional validation: ensure paths are within tmp_path
                tmp_path_str = str(tmp_path.resolve())
                for p in [mo_path, po_path]:
                    if not str(p.resolve()).startswith(tmp_path_str):
                        raise ValueError(f"Unsafe file path detected: {p}")
                subprocess.run(
                    [
                        msgfmt_path,
                        "-o",
                        str(mo_path),
                        str(po_path),
                    ],
                    check=True,
                )
            else:
                raise ValueError("Unsafe command or arguments for msgfmt.")
        else:
            raise ValueError("Invalid .po or .mo file path for msgfmt.")


def test_app_bootstrap_initializes_ui_i18n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    conf_path = _write_cfg(tmp_path)
    _write_streamlit_po(tmp_path)

    # Point app module to tmp conf path
    
    monkeypatch.setattr(app_mod, "CONF_PATH", conf_path, raising=False)

    # Reset session state for isolation
    st.session_state.clear()

    # Act
    app = MLPStreamlitApp()
    app.bootstrap()

    # Assert orchestrators and UI manager present
    if "cfg_orch" not in st.session_state:
        raise AssertionError("'cfg_orch' not found in st.session_state")
    if "msg_orch" not in st.session_state:
        raise AssertionError("'msg_orch' not found in st.session_state")
    if "ui_mm" not in st.session_state:
        raise AssertionError("'ui_mm' not found in st.session_state")

    # Verify UI resolution returns either translated or default
    title = app.get_ui_text(KEY_APP_TITLE)
    if title not in ("Titre UI Test", "mlp — Console (Programmatic Routing)"):
        raise AssertionError(
            f"Unexpected title: {title!r}. Expected one of: 'Titre UI Test', 'mlp — Console (Programmatic Routing)'"
        )
