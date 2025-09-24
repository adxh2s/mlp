# --- imports en tête inchangés ou déjà triés ---

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy import typing as npt

from .config import DLConfig
from .factory import build_model, compile_model

HISTORY_INDEX = False

Float32Array: TypeAlias = npt.NDArray[np.float32]


def get_keras() -> Any:
    try:
        return importlib.import_module("tensorflow.keras")
    except Exception:  # noqa: BLE001
        return None


def _make_callbacks(cfg: DLConfig) -> list[Any]:
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
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def _to_float32_array(x: npt.ArrayLike) -> Float32Array:
    # scipy.sparse
    if hasattr(x, "toarray"):
        x = x.toarray()  # type: ignore[assignment]

    # DataFrame -> ndarray float32
    if isinstance(x, pd.DataFrame):
        df: pd.DataFrame = x
        obj_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt == "object"]
        if obj_cols:
            raise ValueError(f"Colonnes non numériques détectées après preprocess: {obj_cols}")
        # Étape 1: ndarray float64 (annotation explicite pour Pylance)
        df_nd64: npt.NDArray[np.float64] = df.to_numpy(dtype=np.float64, copy=False)
        # Étape 2: conversion stable en float32
        return df_nd64.astype(np.float32, copy=False)

    # Array-like générique -> ndarray float32
    arr32 = np.asarray(x, dtype=np.float32)
    if not np.issubdtype(arr32.dtype, np.number):
        raise ValueError(f"Type non numérique détecté: {arr32.dtype}")
    return arr32


def _to_array(y: npt.ArrayLike) -> Float32Array:
    # Series -> ndarray float32
    if isinstance(y, pd.Series):
        s: pd.Series[Any] = y
        # Étape 1: ndarray float64 (annotation explicite)
        s_nd64: npt.NDArray[np.float64] = s.to_numpy(dtype=np.float64, copy=False)
        # Étape 2: conversion stable en float32
        return s_nd64.astype(np.float32, copy=False)
    # Autres -> ndarray float32
    return np.asarray(y, dtype=np.float32)


def train_dense(
    x_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    x_val: npt.ArrayLike | None,
    y_val: npt.ArrayLike | None,
    cfg: DLConfig,
) -> dict[str, Any]:
    k: Any = get_keras()
    if k is None:
        raise ImportError("TensorFlow/Keras non disponible; installez tensorflow>=2.16,<2.18")

    x_tr: Float32Array = _to_float32_array(x_train)
    y_tr: Float32Array = _to_array(y_train)

    x_v: Float32Array | None = _to_float32_array(x_val) if x_val is not None else None
    y_v: Float32Array | None = _to_array(y_val) if y_val is not None else None

    if cfg.model.input_shape is None:
        n_features = int(x_tr.shape[1])
        cfg.model.input_shape = [n_features]

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

    if x_v is not None and y_v is not None:
        fit_kwargs["validation_data"] = (x_v, y_v)
    elif fit_cfg.validation_split is not None:
        fit_kwargs["validation_split"] = fit_cfg.validation_split

    if fit_cfg.steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = fit_cfg.steps_per_epoch
    if fit_cfg.validation_steps is not None:
        fit_kwargs["validation_steps"] = fit_cfg.validation_steps
    if fit_cfg.shuffle is not None:
        fit_kwargs["shuffle"] = fit_cfg.shuffle

    history_any: Any = model.fit(x=x_tr, y=y_tr, callbacks=callbacks, **fit_kwargs)
    history: dict[str, list[float]] = {
        key: [float(v) for v in vals] for key, vals in history_any.history.items()
    }

    if cfg.export.save_model:
        Path(cfg.export.path).parent.mkdir(parents=True, exist_ok=True)
        model.save(cfg.export.path)

    if cfg.export.save_history_csv:
        Path(cfg.export.save_history_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(cfg.export.save_history_csv, index=HISTORY_INDEX)

    final_metrics = {k2: v[-1] for k2, v in history.items() if v}

    return {
        "summary": summary_text,
        "history": history,
        "final_metrics": final_metrics,
        "model_path": cfg.export.path if cfg.export.save_model else None,
        "history_csv": cfg.export.save_history_csv,
    }
