from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.dl.factory import get_keras
from src.modeling.pipelines.evaluator import PipelineEvaluator


@pytest.mark.skipif(get_keras() is None, reason="TensorFlow/Keras not installed")
def test_evaluator_runs_dl(tmp_path) -> None:
    # Données jouets
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(80, 10)).astype("float32"))
    y = pd.Series((rng.random(80) > 0.5).astype("int32"))

    spec = {
        "name": "dl_dense_auto",
        "enabled": True,
        "automl": {
            "library": "dl",
            "name": "dl_dense_auto",
            "dl": {
                "model": {
                    "type": "sequential",
                    "input_shape": [10],
                    "layers": [{"type": "Dense", "params": {"units": 16, "activation": "relu"}}],
                    "task": "binary",
                    "auto_output": True,
                },
                "compile": {"optimizer": {"name": "adam", "lr": 0.001}, "loss": "auto", "metrics": ["accuracy"]},
                "fit": {"epochs": 1, "batch_size": 16, "validation_split": 0.2, "verbose": 0},
                "export": {
                    "save_model": True,
                    "path": str(tmp_path / "model.keras"),
                    "save_history_csv": str(tmp_path / "hist.csv"),
                },
            },
        },
    }

    evaluator = PipelineEvaluator(out_dir=str(tmp_path), random_state=42, mlflow_enabled=False)
    res = evaluator.evaluate(X, y, spec, cv_cfg={}, global_policy={})
    # Champs DL
    assert res["name"] == "dl_dense_auto"
    assert "artifacts" in res and isinstance(res["artifacts"], list)
    assert "metrics" in res and isinstance(res["metrics"], dict)
    # Un score utilisable si val_accuracy présent
    assert "best_score" in res
