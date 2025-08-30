from __future__ import annotations

from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ..config.schemas import AppConfig


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
