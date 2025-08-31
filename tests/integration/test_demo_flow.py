from __future__ import annotations

from omegaconf import OmegaConf

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator


def test_integration_end_to_end(tmp_outputs):
    cfg = OmegaConf.create(
        {
            "project": {"name": "it_project", "random_state": 42, "output_dir": str(tmp_outputs)},
            "orchestrators": {
                "eda": {"enabled": True, "profile": {"minimal": True}},
                "pipelines": {
                    "enabled": True,
                    "cv": {"cv_folds": 2, "scoring": ["f1"]},
                    "pipelines": [
                        {"name": "rf_small", "steps": {"preprocess": {"imputer": "simple"}, "estimator": {"type": "random_forest", "params": {"n_estimators": 5}}}},
                    ],
                },
                "report": {"enabled": True, "formats": ["md"]},
            },
            "logger": {"backend": "stdlib", "level": "WARNING"},
        }
    )
    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()

    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"]).head(60)
    y = ds.frame["target"].head(60)

    go = GeneralOrchestrator(cfg_mgr)
    out = go.run(X, y)
    arts = out.get("report", {}).get("artifacts", [])
    assert len(arts) >= 1
