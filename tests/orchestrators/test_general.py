from __future__ import annotations

from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator


def test_general_orchestrator_demo_flow(tmp_outputs):
    # Minimal config for a tiny run in tests
    cfg = OmegaConf.create(
        {
            "project": {"name": "test_project", "random_state": 42, "output_dir": str(tmp_outputs)},
            "orchestrators": {
                "eda": {"enabled": True, "profile": {"minimal": True}},
                "pipelines": {
                    "enabled": True,
                    "cv": {"cv_folds": 2, "scoring": ["f1"]},
                    "pipelines": [
                        {"name": "svc_small", "steps": {"preprocess": {"imputer": "simple"}, "estimator": {"type": "svc", "params": {"kernel": ["linear"], "C": [0.1]}}}}
                    ],
                },
                "report": {"enabled": True, "formats": ["md"]},
            },
            "logger": {"backend": "stdlib", "level": "WARNING"},
        }
    )
    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()

    # small demo dataset
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"]).head(40)
    y = ds.frame["target"].head(40)

    go = GeneralOrchestrator(cfg_mgr)
    out = go.run(X, y)
    assert "report" in out
    assert isinstance(out["report"].get("artifacts", []), list)
