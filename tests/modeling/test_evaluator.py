from __future__ import annotations

from src.modeling.evaluator import PipelineEvaluator


def test_evaluator_grid_runs(tmp_outputs, demo_dataset):
    X, y = demo_dataset
    spec = {
        "name": "svc_small",
        "steps": {
            "preprocess": {"imputer": "simple", "scaler": "standard"},
            "estimator": {"type": "svc", "params": {"kernel": ["linear"], "C": [0.1]}},
        },
    }
    ev = PipelineEvaluator(out_dir=str(tmp_outputs), random_state=42, mlflow_enabled=False)
    res = ev.evaluate(X, y, spec, cv_cfg={"cv_folds": 3, "scoring": ["f1"]})
    assert res["name"] == "svc_small"
    assert "best_score" in res
    assert "cv_results_path" in res
