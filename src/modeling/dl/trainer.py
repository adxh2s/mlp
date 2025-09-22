from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DLConfig
from .factory import build_model, compile_model

"""
Boucle d’entraînement DL (Keras): summary, fit+callbacks, export modèle et historique.
"""


# Constantes
HISTORY_INDEX = False


def get_keras() -> Any:
    """
    Charge le module tensorflow.keras dynamiquement.
    Retourne le module Keras (type Any) ou None si indisponible.
    """
    try:
        return importlib.import_module("tensorflow.keras")
    except Exception:  # noqa: BLE001
        return None


def _make_callbacks(cfg: DLConfig) -> list[Any]:
    """Construit la liste des callbacks actifs à partir de la config."""
    k: Any = get_keras()
    if k is None:
        return []
    cb: list[Any] = []
    ec = cfg.callbacks.early_stopping
    if ec.enabled:
        cb.append(
            k.callbacks.EarlyStopping(
                monitor=ec.monitor,
                patience=ec.patience,
                restore_best_weights=ec.restore_best_weights,
                mode=ec.mode,
                min_delta=ec.min_delta,
                verbose=ec.verbose,
            )
        )
    rc = cfg.callbacks.reduce_lr
    if rc.enabled:
        cb.append(
            k.callbacks.ReduceLROnPlateau(
                monitor=rc.monitor,
                factor=rc.factor,
                patience=rc.patience,
                min_lr=rc.min_lr,
                mode=rc.mode,
                verbose=rc.verbose,
                min_delta=rc.min_delta,
                cooldown=rc.cooldown,
            )
        )
    cc = cfg.callbacks.checkpoint
    if cc.enabled:
        Path(cc.filepath).parent.mkdir(parents=True, exist_ok=True)
        cb.append(
            k.callbacks.ModelCheckpoint(
                filepath=cc.filepath,
                monitor=cc.monitor,
                save_best_only=cc.save_best_only,
                save_weights_only=cc.save_weights_only,
                mode=cc.mode,
                verbose=cc.verbose,
            )
        )
    return cb


def _summary_to_string(model: Any) -> str:
    """Capture le summary Keras dans une chaîne."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def train_dense(
    x_train: Any,
    y_train: Any,
    x_val: Any | None,
    y_val: Any | None,
    cfg: DLConfig,
) -> dict[str, Any]:
    """
    Entraîne un MLP dense défini par la config et retourne summary, history, métriques et artefacts.
    """
    k: Any = get_keras()
    if k is None:
        raise ImportError("TensorFlow/Keras non disponible; installez tensorflow>=2.16,<2.18")

    model: Any = build_model(cfg)
    compile_model(model, cfg)

    summary_text = _summary_to_string(model)

    fit_cfg = cfg.fit
    callbacks = _make_callbacks(cfg)

    fit_kwargs: dict[str, Any] = dict(
        epochs=fit_cfg.epochs,
        batch_size=fit_cfg.batch_size,
        verbose=fit_cfg.verbose,
    )
    if x_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (x_val, y_val)
    elif fit_cfg.validation_split is not None:
        fit_kwargs["validation_split"] = fit_cfg.validation_split
    if fit_cfg.steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = fit_cfg.steps_per_epoch
    if fit_cfg.validation_steps is not None:
        fit_kwargs["validation_steps"] = fit_cfg.validation_steps
    if fit_cfg.shuffle is not None:
        fit_kwargs["shuffle"] = fit_cfg.shuffle

    history: Any = model.fit(x=x_train, y=y_train, callbacks=callbacks, **fit_kwargs)
    hist_dict: dict[str, list[float]] = {k2: list(map(float, v)) for k2, v in history.history.items()}

    if cfg.export.save_model:
        Path(cfg.export.path).parent.mkdir(parents=True, exist_ok=True)
        model.save(cfg.export.path)

    if cfg.export.save_history_csv:
        Path(cfg.export.save_history_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist_dict).to_csv(cfg.export.save_history_csv, index=HISTORY_INDEX)

    final_metrics = {k3: v[-1] for k3, v in hist_dict.items() if len(v) > 0}

    return {
        "summary": summary_text,
        "history": hist_dict,
        "final_metrics": final_metrics,
        "model_path": cfg.export.path if cfg.export.save_model else None,
        "history_csv": cfg.export.save_history_csv,
    }
