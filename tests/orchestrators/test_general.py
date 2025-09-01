from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf
from sklearn.datasets import load_breast_cancer

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator


def test_general_orchestrator_with_file(tmp_path):
    # Préparer un petit fichier d'entrée CSV dans data_dir/in_dir
    data_root = tmp_path
    in_dir = data_root / "in"
    out_dir = data_root / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "input.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    # Config minimale avec orchestrator file activé
    cfg = OmegaConf.create(
        {
            "project": {"name": "test_project", "random_state": 42, "output_dir": str(tmp_path)},
            "orchestrators": {
                "file": {
                    "enabled": True,
                    "data_dir": str(data_root),
                    "in_dir": "in",
                    "out_dir": "out",
                    "extensions": [".csv"],
                    "save_input_file": True,
                    "save_input_file_compression": False,
                },
                "eda": {"enabled": True, "profile": {"minimal": True}},
                "pipelines": {
                    "enabled": True,
                    "cv": {"cv_folds": 2, "scoring": ["f1"]},
                    "pipelines": [
                        {
                            "name": "svc_small",
                            "steps": {
                                "preprocess": {"imputer": "simple"},
                                "estimator": {"type": "svc", "params": {"kernel": ["linear"], "C": [0.1]}},
                            },
                        }
                    ],
                },
                "report": {"enabled": True, "formats": ["md"]},
            },
            "logger": {"backend": "stdlib", "level": "WARNING"},
        }
    )

    cfg_mgr = ConfigManager(cfg)
    cfg_mgr.load()

    # Dataset réduit
    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"]).head(30)
    y = ds.frame["target"].head(30)

    go = GeneralOrchestrator(cfg_mgr)
    out = go.run(X, y)

    # Vérifications: sections présentes et file intake effectué
    assert "file" in out
    fsec = out["file"]
    assert fsec.get("found") is True
    assert fsec.get("file", "").endswith("input.csv")
    # saved_copy doit exister dans out_dir (horodaté)
    saved = fsec.get("saved_copy")
    assert saved is None or saved.endswith(".csv")  # copié si save_input_file=True
    assert "eda" in out
    assert "pipelines" in out
    assert "report" in out
    assert isinstance(out["report"].get("artifacts", []), list)
