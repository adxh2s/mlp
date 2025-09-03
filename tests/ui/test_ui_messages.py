from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

import pytest

from src.instrumentation.messages_manager import MessageManager

"""
Tests for UI MessageManager domain (streamlit_app):
- Ensures .po/.mo translation works for "app_title" key.
- Avoids Ruff S603/S607 by resolving absolute msgfmt path.
- Avoids Ruff S101 by explicit checks raising AssertionError.
"""


def _write_streamlit_po(tmp_path: Path) -> Path:
    lcd = tmp_path / "i18n" / "locales" / "fr" / "LC_MESSAGES"
    lcd.mkdir(parents=True, exist_ok=True)
    po = lcd / "streamlit_app.po"
    po.write_text(
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
    return po


def _compile_mo(po: Path) -> Path | None:
    """
    Compile .po to .mo using system msgfmt with absolute path.

    Returns the mo path if compiled, otherwise None (test will be skipped).
    """
    msgfmt_path = which("msgfmt")
    if msgfmt_path is None:
        pytest.skip("msgfmt not available in PATH")

    mo = po.with_suffix(".mo")
    # Entrée contrôlée: msgfmt_path provient de shutil.which, pas d'input utilisateur,
    # et nous passons une liste d'arguments (shell=False). Sûr pour ce test.
    subprocess.run([msgfmt_path, "-o", str(mo), str(po)], check=True)  # noqa: S603
    if not mo.exists():
        raise AssertionError("Expected compiled .mo to exist")
    return mo


def test_ui_message_manager_translation(tmp_path: Path) -> None:
    po = _write_streamlit_po(tmp_path)
    mo = _compile_mo(po)
    if mo is None:  # pragma: no cover
        pytest.skip("msgfmt unavailable")

    mm = MessageManager(tmp_path / "i18n" / "locales", default_locale="fr")
    txt = mm.msg("streamlit_app", "app_title")

    if txt != "Titre UI Test":
        raise AssertionError(f'Expected "Titre UI Test", got "{txt}"')
