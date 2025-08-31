from __future__ import annotations

from src.modeling.pipeline_factory import PipelineFactory


def test_pipeline_factory_build_linear_svc():
    fac = PipelineFactory(random_state=42)
    spec = {
        "name": "linear_pca_svc",
        "steps": {
            "preprocess": {"imputer": "simple", "scaler": "standard"},
            "feature_selection": {"select_k_best": 3},
            "reduction": {"type": "pca", "params": {"n_components": [0.8, 0.9]}},
            "estimator": {"type": "svc", "params": {"kernel": ["linear"], "C": [0.1, 1]}},
        },
    }
    pipe, grid = fac.build(spec)
    assert pipe is not None
    assert any("estimator" in name for name, _ in pipe.steps)
    assert "reduction__n_components" in grid
    assert "estimator__C" in grid
