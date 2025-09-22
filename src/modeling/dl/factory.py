from __future__ import annotations

import importlib
from typing import Any

from src.modeling.dl.consts import (
    LAYER_NAMES,
    MODEL_FUNCTIONAL,
    MODEL_NAME_FUNC,
    MODEL_NAME_SEQ,
    MODEL_SEQUENTIAL,
    TASK_BINARY,
    TASK_MULTICLASS,
)

from .config import DLConfig, LayerSpec

"""
Construction/Compilation Keras depuis config:
- Empilement des couches (séquentiel/fonctionnel).
- Sortie auto (sigmoïde/softmax) si auto_output.
- Mapping optimizer.lr -> learning_rate.
"""


def _get_keras() -> Any:
    """Charge tensorflow.keras dynamiquement pour éviter les erreurs d'import avant installation."""
    try:
        return importlib.import_module("tensorflow.keras")
    except Exception:  # noqa: BLE001
        return None


def _resolve_layer(name: str) -> Any:
    """Retourne keras.layers.<name> ou None si indisponible."""
    k = _get_keras()
    layers_mod: Any = getattr(k, "layers", None) if k is not None else None
    return getattr(layers_mod, name, None) if layers_mod is not None else None


def _build_layers(specs: list[LayerSpec]) -> list[Any]:
    """Instancie les couches Keras depuis la description déclarative."""
    layers: list[Any] = []
    for ls in specs:
        name = ls.type
        if name not in LAYER_NAMES:
            raise ValueError(f"Unsupported layer: {name}")
        layer_cls: Any = _resolve_layer(name)
        if layer_cls is None:
            raise ImportError("TensorFlow/Keras non disponible; installez tensorflow>=2.16,<2.18")
        params: dict[str, Any] = dict(ls.params)
        layers.append(layer_cls(**params))
    return layers


def _is_dense_with_activation(layer: Any) -> bool:
    """Détecte une Dense déjà configurée avec activation."""
    return getattr(layer, "__class__", type("X", (), {})).__name__ == "Dense" and getattr(layer, "activation", None) is not None


def _maybe_append_output(cfg: DLConfig, layers: list[Any]) -> list[Any]:
    """Ajoute automatiquement une sortie adaptée si auto_output et pas déjà fournie."""
    if not cfg.model.auto_output:
        return layers
    dense_cls = _resolve_layer("Dense")
    if not dense_cls:
        return layers
    if layers and _is_dense_with_activation(layers[-1]):
        return layers
    task = cfg.model.task
    n_classes = cfg.model.n_classes
    out_layers = list(layers)
    if task == TASK_BINARY:
        out_layers.append(dense_cls(1, activation="sigmoid"))
    elif task == TASK_MULTICLASS and n_classes and n_classes > 1:
        out_layers.append(dense_cls(n_classes, activation="softmax"))
    return out_layers


def build_model(cfg: DLConfig) -> Any:
    """Construit un modèle Keras conforme à la config (séquentiel ou fonctionnel)."""
    k = _get_keras()
    if k is None:
        raise ImportError("TensorFlow/Keras non disponible; installez tensorflow>=2.16,<2.18")
    m = cfg.model
    if m.type == MODEL_SEQUENTIAL:
        if not m.input_shape:
            raise ValueError("input_shape is required for sequential models")
        model: Any = k.Sequential(name=MODEL_NAME_SEQ)
        input_layer = _resolve_layer("Input")
        if input_layer is None:
            raise ImportError("Keras.layers.Input non disponible")
        model.add(input_layer(shape=tuple(m.input_shape)))
        for layer in _maybe_append_output(cfg, _build_layers(m.layers)):
            model.add(layer)
        return model

    if m.type != MODEL_FUNCTIONAL:
        raise ValueError(f"Unknown model type: {m.type}")

    if not m.input_shape:
        raise ValueError("input_shape is required for functional models")
    input_layer = _resolve_layer("Input")
    if input_layer is None:
        raise ImportError("Keras.layers.Input non disponible")
    inputs: Any = input_layer(shape=tuple(m.input_shape), name="inputs")
    x: Any = inputs
    for layer in _maybe_append_output(cfg, _build_layers(m.layers)):
        x = layer(x)
    return k.Model(inputs=inputs, outputs=x, name=MODEL_NAME_FUNC)


def compile_model(model: Any, cfg: DLConfig) -> None:
    """Compile le modèle avec optimizer (lr supporté), loss (auto) et metrics."""
    k = _get_keras()
    if k is None:
        raise ImportError("TensorFlow/Keras non disponible; installez tensorflow>=2.16,<2.18")

    cc = cfg.compile
    # Optimizer
    opt = cc.optimizer
    if isinstance(opt, str):
        optimizer: Any = opt
    else:
        name = opt.get("name", "adam")
        params = {k2: v for k2, v in opt.items() if k2 != "name"}
        if "lr" in params and "learning_rate" not in params:
            params["learning_rate"] = params.pop("lr")
        optimizer = k.optimizers.get({"class_name": name, "config": params})

    # Loss
    loss: str | dict[str, Any] = cc.loss
    if isinstance(loss, str) and loss == "auto":
        loss = "binary_crossentropy" if cfg.model.task == TASK_BINARY else "sparse_categorical_crossentropy"

    # Metrics
    metrics: list[str | dict[str, Any]] = list(cc.metrics) if cc.metrics else []

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
