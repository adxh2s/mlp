from __future__ import annotations

import pytest

from src.modeling.dl.config import DLConfig, LayerSpec
from src.modeling.dl.factory import build_model, get_keras


@pytest.mark.skipif(get_keras() is None, reason="TensorFlow/Keras not installed")
def test_build_sequential_binary_auto_output() -> None:
    cfg = DLConfig(
        model={
            "type": "sequential",
            "input_shape": [10],
            "layers": [LayerSpec(type="Dense", params={"units": 16, "activation": "relu"}).model_dump()],
            "task": "binary",
            "auto_output": True,
        }
    )
    model = build_model(cfg)
    last = model.layers[-1]
    # Dense + sigmoid
    assert last.__class__.__name__ == "Dense"
    assert getattr(last, "activation").__name__ == "sigmoid"
