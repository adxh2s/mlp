
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.config import ConfigOrchestrator

"""
Tests for ConfigOrchestrator:
- Loads validated AppConfig through ConfigManager
- Initializes a shared LoggerManager
- Emits localized events (config_ready / config_error)
"""


def _build_cfg(minimal: bool = True) -> Any:
    """Build a minimal OmegaConf DictConfig for tests."""
    base: dict[str, Any] = {
        "project": {"name": "demo", "random_state": 42, "output_dir": "outputs"},
        "orchestrators": {
            "config": {"enabled": True},
            "messages": {
                "enabled": True,
                "locale": "fr",
                "locales_dir": "i18n/locales",
                "domains": ["general", "config"],
            },
        },
        "logger": {
            "backend": "stdlib",
            "app_name": "mlp",
            "level": "INFO",
            "json_mode": False,
            "file_path": None,
            "default_fields": {"app": "mlp", "env": "test"},
        },
    }
    return OmegaConf.create(base)


def test_config_orchestrator_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    cfg = _build_cfg()
    cfg_mgr = ConfigManager(cfg)

    # Fake project root for config manager (so that any file path resolutions are under tmp)
    monkeypatch.setattr(
        type(cfg_mgr),
        "project_root",
        property(lambda self: tmp_path),  # override property
        raising=False,
    )

    # Spy logger manager configure to ensure it's called
    with patch("src.instrumentation.logger_factory.build_logger_manager") as blm:
        lm = MagicMock()
        blm.return_value = lm

        # Act
        orch = ConfigOrchestrator(cfg_mgr)

        # Assert
        if orch.get_app_config().project.name != "demo":
            raise AssertionError("Expected project name to be 'demo'")
        if orch.get_output_dir() != "outputs":
            raise AssertionError("Expected output_dir to be 'outputs'")
        blm.assert_called_once()
        lm.configure.assert_called_once()


def test_config_orchestrator_emits_error_on_invalid_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: invalid config (missing project section)
    bad = OmegaConf.create({"orchestrators": {}, "logger": {}})
    cfg_mgr = ConfigManager(bad)

    # Intercept MessageOrchestrator.emit to capture error emission
    events: list[dict[str, str]] = []

    def fake_emit(domain: str, event: str, level: str = "info", **fields: Any) -> None:  # noqa: D401
        events.append({"domain": domain, "event": event, "level": level, **{k: str(v) for k, v in fields.items()}})

    with patch("src.orchestrators.config.MessageOrchestrator") as MM:
        instance = MagicMock()
        instance.emit.side_effect = fake_emit
        MM.return_value = instance

        # Act / Assert: constructor should raise due to invalid cfg
        with pytest.raises(ValueError):
            ConfigOrchestrator(cfg_mgr)

    # Verify an error event was emitted
    if not any(e for e in events if e["level"] == "error"):
        raise ValueError("Expected at least one error event to be emitted")
