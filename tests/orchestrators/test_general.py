from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from sklearn.datasets import load_breast_cancer

from src.instrumentation.config_manager import ConfigManager
from src.orchestrators.general import GeneralOrchestrator


def test_general_orchestrator_with_file(tmp_path: Path) -> None:
    # Préparer un petit fichier d'entrée CSV dans data_dir/in_dir
    data_root: Path = tmp_path
    in_dir: Path = data_root / "in"
    out_dir: Path = data_root / "out"
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
    x = ds.frame.drop(columns=["target"]).head(30)
    y = ds.frame["target"].head(30)

    go = GeneralOrchestrator(cfg_mgr)
    out = go.run(x, y)

    # Vérifications avec exceptions explicites (pas d'assert)
    if "file" not in out:
        raise AssertionError("'file' section missing in orchestrator output")  # noqa: S101

    fsec = out["file"]
    if not fsec.get("found"):
        raise AssertionError("file intake did not mark 'found' as True")  # noqa: S101

    if not fsec.get("file", "").endswith("input.csv"):
        raise AssertionError("expected 'file' to end with 'input.csv'")  # noqa: S101

    saved = fsec.get("saved_copy")
    if not (saved is None or saved.endswith(".csv")):
        raise AssertionError("saved_copy should be None or end with .csv")  # noqa: S101

    if "eda" not in out:
        raise AssertionError("'eda' section missing in orchestrator output")  # noqa: S101
    if "pipelines" not in out:
        raise AssertionError("'pipelines' section missing in orchestrator output")  # noqa: S101
    if "report" not in out:
        raise AssertionError("'report' section missing in orchestrator output")  # noqa: S101
    if not isinstance(out["report"].get("artifacts", []), list):
        raise AssertionError("'report.artifacts' should be a list")  # noqa: S101
