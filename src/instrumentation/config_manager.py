"""Configuration manager for Hydra/OmegaConf → Pydantic models and logging settings.

This module isolates Hydra-specific logic (OmegaConf) from service orchestrators,
so services receive typed settings without depending on Hydra or OmegaConf directly.
It also centralizes environment overrides (e.g., MLP_LOG_FILE) and project-root
path normalization to keep portability between local runs and Docker/CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ..config.schemas import AppConfig, LoggerSettings


class ConfigManager:
    """Load and validate Hydra configuration and provide typed accessors."""

    ERR_INVALID_CFG = "Configuration invalid"
    KEY_PROJECT = "project"
    KEY_ORCHESTRATORS = "orchestrators"

    def __init__(self, hydra_cfg: DictConfig) -> None:
        """Initialize with a Hydra/OmegaConf DictConfig."""
        self._hydra_cfg = hydra_cfg
        self._pyd_model: AppConfig | None = None
        self._raw: dict[str, Any] | None = None
        self._project_root: Path | None = None

    def load(self) -> AppConfig:
        """Resolve and validate the configuration into AppConfig."""
        raw = OmegaConf.to_container(self._hydra_cfg, resolve=True)
        try:
            self._pyd_model = AppConfig(**raw)  # type: ignore[arg-type]
            self._raw = raw  # type: ignore[assignment]
        except ValidationError as exc:  # pragma: no cover
            raise ValueError(f"{self.ERR_INVALID_CFG}: {exc}") from exc
        return self._pyd_model

    @property
    def model(self) -> AppConfig:
        """Return the validated AppConfig, loading it if needed."""
        if self._pyd_model is None:
            return self.load()
        return self._pyd_model

    @property
    def raw(self) -> dict[str, Any]:
        """Return the resolved raw dict config for ad‑hoc access."""
        if self._raw is None:
            _ = self.load()
        return self._raw or {}

    @property
    def project_root(self) -> str:
        """Detect a stable project root (pyproject.toml/.git), fallback to parent of CWD."""
        if self._project_root is not None:
            return str(self._project_root)
        start = Path.cwd()
        for up in [start, *start.parents]:
            if (up / "pyproject.toml").exists() or (up / ".git").exists():
                self._project_root = up
                return str(up)
        self._project_root = start.parent
        return str(self._project_root)

    def make_logs_file_path(self, name: str) -> str:
        """Build a default file path under <project_root>/logs/<name>."""
        base = Path(self.project_root)
        return str(base / "logs" / name)

    def build_logger_settings(self) -> LoggerSettings:
        """Compose LoggerSettings from Hydra YAML + env overrides (no stdlib wiring here).

        Resolution order (by design):
        1) Hydra YAML under `orchestrators.logger` (primary source of truth).
        2) Environment override for `file_path` (MLP_LOG_FILE) for container/CI portability.
        3) Reasonable defaults for local runs (e.g., logs/app.log).

        Handlers/root_handlers (if present) are forwarded as-is so the logging
        manager can build dictConfig dynamically without duplicating YAML parsing.
        """
        raw = self.raw
        # Prefer orchestrators.logger, fallback to root-level logger if present.
        orch = raw.get(self.KEY_ORCHESTRATORS, {}) if isinstance(raw, dict) else {}
        logger_raw: dict[str, Any] = {}
        if isinstance(orch, dict):
            lr = orch.get("logger")
            if isinstance(lr, dict):
                logger_raw = lr
        if not logger_raw and isinstance(raw, dict):
            lr = raw.get("logger")
            if isinstance(lr, dict):
                logger_raw = lr

        project = raw.get(self.KEY_PROJECT, {}) if isinstance(raw, dict) else {}
        app_name = (
            logger_raw.get("app_name")
            or (project.get("name") if isinstance(project, dict) else None)
            or "mlp"
        )

        # Environment override for file path (Docker/CI/Compose)
        import os
        env_path = os.getenv("MLP_LOG_FILE")
        file_path = env_path or logger_raw.get("file_path") or self.make_logs_file_path("app.log")

        settings_kwargs: dict[str, Any] = {
            "backend": logger_raw.get("backend", "stdlib"),
            "app_name": app_name,
            "level": logger_raw.get("level", "INFO"),
            "json_mode": logger_raw.get("json_mode", False),
            "file_path": file_path,
            "file_max_bytes": logger_raw.get("file_max_bytes", 5 * 1024 * 1024),
            "file_backup_count": logger_raw.get("file_backup_count", 3),
            "uvicorn_noise_filter": logger_raw.get("uvicorn_noise_filter", True),
            "default_fields": logger_raw.get("default_fields", {}),
        }
        # Advanced routing (optional) stays declarative in YAML.
        if "handlers" in logger_raw:
            settings_kwargs["handlers"] = logger_raw.get("handlers")
        if "root_handlers" in logger_raw:
            settings_kwargs["root_handlers"] = logger_raw.get("root_handlers")

        return LoggerSettings(**settings_kwargs)
