from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt

from src.modeling.dl.consts import DEF_CKPT_PATH, DEF_MODEL_PATH, DEF_MONITOR

"""
Schémas DL déclaratifs (Keras): modèle, compilation, fit, callbacks, export.
"""

class LayerSpec(BaseModel):
    """Couche déclarative: type + paramètres (ex: Dense, Dropout)."""
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


def _empty_layers() -> list[LayerSpec]:
    """Fabrique typée pour éviter list[Unknown] côté Pylance."""
    return []


class ModelConfig(BaseModel):
    """Description du modèle Keras et de la tâche."""
    type: Literal["sequential", "functional"] = "sequential"
    input_shape: list[int] | None = None
    layers: list[LayerSpec] = Field(default_factory=_empty_layers)
    task: Literal["binary", "multiclass"] | None = None
    n_classes: int | None = None
    auto_output: bool = True


class CompileConfig(BaseModel):
    """Compilation: optimizer (string ou dict avec lr), loss (auto), metrics."""
    optimizer: str | dict[str, Any] = "adam"
    loss: str | dict[str, Any] = "auto"
    metrics: list[str | dict[str, Any]] = Field(default_factory=lambda: ["accuracy"])


class FitConfig(BaseModel):
    """Paramètres d’entraînement fit()."""
    epochs: PositiveInt = 20
    batch_size: PositiveInt = 32
    validation_split: float | None = None
    verbose: int = 1
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    shuffle: bool | None = None


class EarlyStoppingCfg(BaseModel):
    """Callback EarlyStopping."""
    enabled: bool = False
    monitor: str = DEF_MONITOR
    patience: int = 10
    restore_best_weights: bool = True
    mode: str = "auto"
    min_delta: float = 0.0
    verbose: int = 0


class ReduceLRCfg(BaseModel):
    """Callback ReduceLROnPlateau."""
    enabled: bool = False
    monitor: str = DEF_MONITOR
    factor: float = 0.1
    patience: int = 10
    min_lr: float = 0.0
    mode: str = "auto"
    verbose: int = 0
    min_delta: float = 1e-4
    cooldown: int = 0


class CheckpointCfg(BaseModel):
    """Callback ModelCheckpoint."""
    enabled: bool = False
    filepath: str = DEF_CKPT_PATH
    monitor: str = DEF_MONITOR
    save_best_only: bool = True
    save_weights_only: bool = False
    mode: str = "auto"
    verbose: int = 0


class CallbacksConfig(BaseModel):
    """Agrégat des callbacks usuels."""
    early_stopping: EarlyStoppingCfg = Field(default_factory=EarlyStoppingCfg)
    reduce_lr: ReduceLRCfg = Field(default_factory=ReduceLRCfg)
    checkpoint: CheckpointCfg = Field(default_factory=CheckpointCfg)


class ExportConfig(BaseModel):
    """Options d’export: modèle et historique."""
    save_model: bool = True
    path: str = DEF_MODEL_PATH
    save_history_csv: str | None = None


class DLConfig(BaseModel):
    """Config DL racine."""
    backend: Literal["keras"] = "keras"
    model: ModelConfig = Field(default_factory=ModelConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
