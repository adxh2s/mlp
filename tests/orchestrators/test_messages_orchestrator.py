
from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.messages import MessageOrchestrator

"""
Tests for MessageOrchestrator:
- Loads locale config and resolves messages via gettext .mo
- Emits structured logs (event + msg + domain + fields)
"""


def _write_po(tmp_path: Path, domain: str) -> Path:
    """Write a minimal French PO file for a domain and return path."""
    lcd = tmp_path / "i18n" / "locales" / "fr" / "LC_MESSAGES"
    lcd.mkdir(parents=True, exist_ok=True)
    po = lcd / f"{domain}.po"
    po.write_text(
        'msgid ""\n'
        'msgstr ""\n'
        '"Project-Id-Version: mlp 1.0\\n"\n'
        '"Language: fr\\n"\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        "\n"
        'msgid "unit_test_event"\n'
        'msgstr "Événement de test {value}"\n',
        encoding="utf-8",
    )
    return po


def _compile_mo(po: Path) -> Path:
    """Compile .po to .mo using msgfmt if available; skip if not installed."""
    mo = po.with_suffix(".mo")
    # Try system msgfmt; if missing, skip test
    if which("msgfmt") is None:
        pytest.skip("msgfmt not available in PATH")


    subprocess.run(["msgfmt", "-o", str(mo), str(po)], check=True)
    if not mo.exists():
        raise RuntimeError(f"Failed to compile .mo file: {mo}")
    return mo


def _build_cfg(locales_dir: str, domains: list[str]) -> Any:
    """Build OmegaConf with messages orchestrator section."""
    base = {
        "project": {"name": "demo", "random_state": 42, "output_dir": "outputs"},
        "orchestrators": {
            "messages": {
                "enabled": True,
                "locale": "fr",
                "locales_dir": locales_dir,
                "domains": domains,
            }
        },
        "logger": {
            "backend": "stdlib",
            "app_name": "mlp",
            "level": "INFO",
            "json_mode": True,  # JSON formatter path for assertions
            "file_path": None,
            "default_fields": {"app": "mlp", "env": "test"},
        },
    }
    return OmegaConf.create(base)


def test_message_orchestrator_translate_and_emit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange: create minimal PO/MO for "general" domain
    po = _write_po(tmp_path, "general")
    _compile_mo(po)

    cfg = _build_cfg(locales_dir="i18n/locales", domains=["general"])
    cfg_mgr = ConfigManager(cfg)

    # Force project_root to tmp_path so locales_dir resolves under tmp
    type(cfg_mgr).project_root = property(lambda self: tmp_path)  # type: ignore[assignment]

    # Use real stdlib logger manager (JSON mode) to capture stdout/stderr
    orch = MessageOrchestrator(cfg_mgr)

    # Act: translate() and emit()
    text = orch.translate("general", "unit_test_event", value=123)
    orch.emit("general", "unit_test_event", value=123)

    # Assert: translation occurred and log emitted in JSON
    if "Événement de test 123" not in text:
        raise AssertionError('"Événement de test 123" not found in text')

    out = capsys.readouterr().err + capsys.readouterr().out  # stdlib logs to stderr by default
    # Normalize by checking presence of structured fields; cannot rely on ordering
    if not ('"event": "unit_test_event"' in out or "unit_test_event" in out):
        raise AssertionError('"event": "unit_test_event" or "unit_test_event" not found in output')
    if not ('"domain": "general"' in out or "general" in out):
        raise AssertionError('"domain": "general" or "general" not found in output')
    if "Événement de test 123" not in out:
        raise AssertionError('"Événement de test 123" not found in output')


def test_message_orchestrator_missing_domain_graceful(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange: no PO/MO present for "unknown" domain; fallback should return key
    cfg = _build_cfg(locales_dir="i18n/locales", domains=["unknown"])
    cfg_mgr = ConfigManager(cfg)
    type(cfg_mgr).project_root = property(lambda self: tmp_path)  # type: ignore[assignment]

    orch = MessageOrchestrator(cfg_mgr)
    # Act
    text = orch.translate("unknown", "unit_test_event", value=1)
    orch.emit("unknown", "unit_test_event", value=1)

    # Assert: fallback returns key as message template (no translation)
    if text not in ("unit_test_event", "unit_test_event"):
        raise AssertionError("Expected text to be 'unit_test_event'")

    out = capsys.readouterr().err + capsys.readouterr().out
    if "unit_test_event" not in out:
        raise AssertionError("'unit_test_event' not found in output")
