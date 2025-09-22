from __future__ import annotations

import numpy as np
import pytest

from src.modeling.dl.config import DLConfig, LayerSpec
from src.modeling.dl.trainer import train_dense
from src.modeling.dl.factory import get_keras


@pytest.mark.skipif(get_keras() is None, reason="TensorFlow/Keras not installed")
def test_train_dense_returns_metrics_and_artifacts(tmp_path) -> None:
    # Données jouets binaire
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 10)).astype("float32")
    y = (rng.random(64) > 0.5).astype("int32")

    out_model = tmp_path / "model.keras"
    out_hist = tmp_path / "hist.csv"

    cfg = DLConfig(
        model={
            "type": "sequential",
            "input_shape": [10],
            "layers": [
                LayerSpec(type="Dense", params={"units": 16, "activation": "relu"}).model_dump(),
                LayerSpec(type="Dropout", params={"rate": 0.1}).model_dump(),
            ],
            "task": "binary",
            "auto_output": True,
        },
        compile={"optimizer": {"name": "adam", "lr": 0.001}, "loss": "auto", "metrics": ["accuracy"]},
        fit={"epochs": 1, "batch_size": 16, "validation_split": 0.2, "verbose": 0},
        export={"save_model": True, "path": str(out_model), "save_history_csv": str(out_hist)},
    )

    out = train_dense(x[:48], y[:48], x[48:], y[48:], cfg)
    assert "summary" in out and isinstance(out["summary"], str)
    assert "history" in out and isinstance(out["history"], dict)
    assert "final_metrics" in out and isinstance(out["final_metrics"], dict)
    assert out["model_path"] == str(out_model)
    assert out["history_csv"] == str(out_hist)
