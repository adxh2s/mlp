from __future__ import annotations

import importlib
from typing import Any

from .config import DLConfig, LayerSpec
from .consts import (
    MODEL_FUNCTIONAL,
    MODEL_NAME_FUNC,
    MODEL_NAME_SEQ,
    MODEL_SEQUENTIAL,
    TASK_BINARY,
    TASK_MULTICLASS,
)

"""
Fabrique Keras (build/compile) à partir d’une config déclarative (DLConfig).

- Construction séquentielle/fonctionnelle depuis LayerSpec.
- Injection automatique de input_shape sur la première couche si absent.
- Ajout automatique d’une couche de sortie (auto_output) adaptée à la tâche:
  * Binary: Dense(1, activation='sigmoid')
  * Multiclass: Dense(n_classes, activation='softmax')
- Normalisation de l’optimizer (string ou dict) et sélection de la loss si "auto".
"""


def _get_keras() -> Any:
    """
    Charge tensorflow.keras dynamiquement.

    Retour:
        Module tensorflow.keras (Any) si disponible, sinon None.
    """
    try:
        return importlib.import_module("tensorflow.keras")
    except Exception:  # noqa: BLE001
        return None


def _resolve_layer(name: str) -> Any:
    """
    Retourne la classe de couche Keras (e.g., Dense, Dropout) par son nom.

    Args:
        name: Nom de la classe de couche Keras (e.g., "Dense").

    Retour:
        La classe Keras.layers.<name> si disponible, sinon None.
    """
    k = _get_keras()
    layers_mod: Any = getattr(k, "layers", None) if k is not None else None
    return getattr(layers_mod, name, None) if layers_mod is not None else None


def _last_dense_signature(layer: Any) -> tuple[int | None, str | None]:
    """
    Extrait (units, activation_name) pour une couche Dense.

    Args:
        layer: Instance Keras de couche.

    Retour:
        (units, activation_name) ou (None, None) si ce n'est pas une Dense.
    """
    if getattr(layer, "__class__", type("X", (), {})).__name__ != "Dense":
        return None, None
    units = getattr(layer, "units", None)
    act = getattr(layer, "activation", None)
    act_name = getattr(act, "__name__", None) if act is not None else None
    return units, act_name


def _maybe_append_output(cfg: DLConfig, layers: list[Any]) -> list[Any]:
    """
    Ajoute une couche de sortie adaptée à la tâche si auto_output est actif
    et si la dernière couche n'est pas déjà une sortie conforme.

    Règles:
      - Binary: Dense(1, 'sigmoid')
      - Multiclass: Dense(n_classes, 'softmax') si n_classes > 1

    Args:
        cfg: Configuration DL (tâche, n_classes, auto_output).
        layers: Liste de couches construites (sans couche de sortie explicite).

    Retour:
        Liste éventuellement augmentée d'une couche de sortie.
    """
    if not cfg.model.auto_output:
        return layers

    dense_cls = _resolve_layer("Dense")
    if not dense_cls:
        return layers

    task = cfg.model.task
    n_classes = cfg.model.n_classes
    out_layers = list(layers)

    # Vérifie si la dernière couche est déjà une vraie sortie
    if out_layers:
        units, act_name = _last_dense_signature(out_layers[-1])
        if task == TASK_BINARY and units == 1 and act_name == "sigmoid":
            return out_layers
        if task == TASK_MULTICLASS and n_classes and n_classes > 1 and units == n_classes and act_name == "softmax":
            return out_layers

    # Ajout automatique si nécessaire
    if task == TASK_BINARY:
        out_layers.append(dense_cls(1, activation="sigmoid"))
    elif task == TASK_MULTICLASS and n_classes and n_classes > 1:
        out_layers.append(dense_cls(n_classes, activation="softmax"))

    return out_layers


def _build_layers(layer_specs: list[LayerSpec], input_shape: list[int] | None) -> list[Any]:
    """
    Construit les instances Keras pour chaque LayerSpec, en injectant input_shape
    sur la première couche si absent dans ses paramètres.

    Args:
        layer_specs: Spécification des couches (type + params).
        input_shape: Shape d'entrée [n_features] ou None.

    Retour:
        Liste de couches Keras instanciées.
    """
    built: list[Any] = []
    for idx, spec in enumerate(layer_specs):
        cls = _resolve_layer(spec.type)
        if cls is None:
            raise ValueError(f"Couche inconnue ou non disponible: {spec.type}")
        params: dict[str, Any] = dict(spec.params or {})
        # Injection input_shape pour la première couche si non fourni
        if idx == 0 and input_shape and "input_shape" not in params:
            params["input_shape"] = tuple(int(x) for x in input_shape)
        built.append(cls(**params))
    return built


def _get_optimizer_factory(optim_mod: Any, name: str) -> Any | None:
    """
    Retourne le constructeur d'optimizer (classe/fabrique) si disponible
    dans keras.optimizers, en testant le nom en minuscules puis capitalisé.

    Args:
        optim_mod: Module keras.optimizers.
        name: Nom logique de l'optimizer (e.g., "adam").

    Retour:
        Fabrique/Classe d'optimizer ou None si introuvable.
    """
    if hasattr(optim_mod, name):
        return getattr(optim_mod, name)
    cap = name.capitalize()
    if hasattr(optim_mod, cap):
        return getattr(optim_mod, cap)
    return None


def _make_optimizer(optimizer_cfg: str | dict[str, Any]) -> Any:
    """
    Crée un optimizer Keras depuis:
      - une chaîne (e.g., "adam"),
      - un dict (e.g., {"name": "adam", "lr": 0.001}), en mappant lr -> learning_rate.

    Args:
        optimizer_cfg: Chaîne ou dict de configuration optimizer.

    Retour:
        Instance d'optimizer Keras.
    """
    k = _get_keras()
    if k is None:
        raise ImportError("tensorflow.keras indisponible")
    optim_mod: Any = getattr(k, "optimizers", None)
    if optim_mod is None:
        raise ImportError("tensorflow.keras.optimizers indisponible")

    # Normaliser en (factory, params)
    if isinstance(optimizer_cfg, str):
        name = optimizer_cfg.strip().lower()
        factory = _get_optimizer_factory(optim_mod, name)
        if factory is None:
            # Fallback sûr et explicite (évite getattr avec constante)
            return optim_mod.Adam()
        return factory()

    # dict[str, Any]
    name: str = str(optimizer_cfg.get("name", "adam")).strip().lower()
    params: dict[str, Any] = dict(optimizer_cfg)
    params.pop("name", None)
    if "lr" in params and "learning_rate" not in params:
        params["learning_rate"] = params.pop("lr")

    factory = _get_optimizer_factory(optim_mod, name)
    if factory is None:
        return optim_mod.Adam(**params)
    return factory(**params)


def _auto_loss(cfg: DLConfig) -> str:
    """
    Détermine automatiquement la loss:
      - binary -> 'binary_crossentropy'
      - multiclass -> 'sparse_categorical_crossentropy' (cible y attendue 1D entiers)

    Args:
        cfg: Configuration DL (tâche).

    Retour:
        Nom de la loss.
    """
    task = cfg.model.task
    if task == TASK_BINARY:
        return "binary_crossentropy"
    return "sparse_categorical_crossentropy"


def build_model(cfg: DLConfig) -> Any:
    """
    Construit le modèle Keras (Sequential ou Functional) à partir de cfg.

    - Applique l'injection input_shape sur la première couche si nécessaire.
    - Ajoute automatiquement une couche de sortie si auto_output est actif.

    Args:
        cfg: DLConfig contenant model.type, input_shape, layers, task, n_classes.

    Retour:
        Modèle Keras non compilé.
    """
    k = _get_keras()
    if k is None:
        raise ImportError("tensorflow.keras indisponible")

    model_type = (cfg.model.type or MODEL_SEQUENTIAL).lower()
    layers = _build_layers(cfg.model.layers or [], cfg.model.input_shape)
    layers = _maybe_append_output(cfg, layers)

    if model_type == MODEL_FUNCTIONAL:
        # Entrée explicite pour fonctionnel
        inputs = k.Input(shape=tuple(int(x) for x in (cfg.model.input_shape or [])))
        x = inputs
        for lyr in layers:
            x = lyr(x)
        model = k.Model(inputs=inputs, outputs=x, name=MODEL_NAME_FUNC)
        return model

    # Séquentiel par défaut
    model = k.Sequential(layers=layers, name=MODEL_NAME_SEQ)
    return model


def compile_model(model: Any, cfg: DLConfig) -> Any:
    """
    Compile le modèle avec optimizer / loss / metrics.

    Args:
        model: Modèle Keras à compiler.
        cfg: DLConfig contenant la config de compilation.

    Retour:
        Modèle Keras compilé.
    """
    k = _get_keras()
    if k is None:
        raise ImportError("tensorflow.keras indisponible")

    comp = cfg.compile
    optimizer = _make_optimizer(getattr(comp, "optimizer", "adam"))
    loss_cfg = getattr(comp, "loss", "auto")
    loss = _auto_loss(cfg) if (loss_cfg is None or str(loss_cfg).lower() == "auto") else loss_cfg
    metrics = list(getattr(comp, "metrics", []) or [])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
