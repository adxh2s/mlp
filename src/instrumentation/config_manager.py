# src/instrumentation/config_manager.py
from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ..config.schemas import AppConfig, LoggerSettings

class ConfigManager:
    """Load and validate OmegaConf (Hydra) config into Pydantic models."""
    ERR_INVALID_CFG = "Configuration invalid"
    KEY_PROJECT = "project"
    KEY_ORCHESTRATORS = "orchestrators"

    def __init__(self, hydra_cfg: DictConfig) -> None:
        """Initialize with a Hydra/OmegaConf DictConfig."""
        self._hydra_cfg = hydra_cfg
        self._pyd_model: Optional[AppConfig] = None
        self._raw: Optional[Dict[str, Any]] = None
        self._project_root: Optional[Path] = None

    def load(self) -> AppConfig:
        """Resolve and validate the configuration, returning AppConfig."""
        raw = OmegaConf.to_container(self._hydra_cfg, resolve=True)
        try:
            self._pyd_model = AppConfig(**raw)  # type: ignore[arg-type]
        except ValidationError as exc:
            raise ValueError(f"{self.ERR_INVALID_CFG}: {exc}") from exc
        self._raw = raw  # type: ignore[assignment]
        return self._pyd_model

    @property
    def model(self) -> AppConfig:
        """Return the validated Pydantic model, loading if needed."""
        return self._pyd_model or self.load()

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the raw resolved dictionary form."""
        return self._raw or OmegaConf.to_container(self._hydra_cfg, resolve=True)  # type: ignore[return-value]

    @property
    def project_root(self) -> Path:
        """Best-effort pour déduire la racine du projet."""
        if self._project_root is not None:
            return self._project_root
        # Heuristique: remonter jusqu'à trouver pyproject.toml ou .git
        here = Path(__file__).resolve()
        cand = here
        for _ in range(8):
            if (cand / "pyproject.toml").exists() or (cand / ".git").exists():
                self._project_root = cand
                break
            cand = cand.parent
        else:
            # Fallback raisonnable: deux niveaux au-dessus
            self._project_root = here.parents[2]
        return self._project_root

    def make_logs_file_path(self, filename: str = "app.log") -> str:
        logs_dir = self.project_root / "logs"
        return str((logs_dir / filename).resolve())

    def build_logger_settings(self) -> LoggerSettings:
        # On récupère la section logger de la config brute si présente
        raw = self.raw
        logger_raw: Dict[str, Any] = raw.get("logger", {}) if isinstance(raw, dict) else {}
        # Si file_path absent → on pointe sur <racine>/logs/app.log
        file_path = logger_raw.get("file_path") or self.make_logs_file_path("app.log")
        # App name: project.name si dispo
        app_name = (
            logger_raw.get("app_name")
            or (raw.get("project", {}) or {}).get("name")
            or "mlp"
        )
        return LoggerSettings(
            backend=logger_raw.get("backend", "stdlib"),
            app_name=app_name,
            level=logger_raw.get("level", "INFO"),
            json_mode=logger_raw.get("json_mode", False),
            file_path=file_path,
            file_max_bytes=logger_raw.get("file_max_bytes", 5 * 1024 * 1024),
            file_backup_count=logger_raw.get("file_backup_count", 3),
            uvicorn_noise_filter=logger_raw.get("uvicorn_noise_filter", True),
            default_fields=logger_raw.get("default_fields", {}),
        )
