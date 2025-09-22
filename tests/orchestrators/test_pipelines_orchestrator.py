from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.orchestrators.pipelines import PipelineOrchestrator


class DummyMsg:
    def emit(self, domain, event, **kwargs):
        # no-op: structure compatible avec l'orchestrateur
        return


def _svc_spec(name="svc_grid", ct_enabled=True):
    return {
        "name": name,
        "enabled": True,
        "steps": {
            "preprocess": {
                "enabled": True,
                "imputer": "simple",
                "column_transformer": {"enabled": ct_enabled, "policy": "auto"},
            },
            "estimator": {
                "type": "sklearn.svm.SVC",
                "params": {"kernel": ["linear"], "C": [0.1, 1.0]},
            },
        },
    }


def _random_spec():
    return {
        "name": "svc_random",
        "enabled": True,
        "steps": {
            "preprocess": {
                "enabled": True,
                "imputer": "simple",
                "column_transformer": {"enabled": True, "policy": "auto"},
            },
            "estimator": {
                "type": "sklearn.svm.SVC",
                "params": {"kernel": ["rbf"]},
                "distributions": {
                    "C": {"dist": "loguniform", "low": 1e-3, "high": 1e2},
                    "gamma": {"dist": "loguniform", "low": 1e-4, "high": 1e-1},
                },
            },
        },
    }


def _lazy_spec():
    return {
        "name": "lazy_cls",
        "enabled": True,
        "automl": {
            "library": "lazypredict",
            "name": "lazy_default",
            "lazy": {"test_size": 0.2, "top_n": 5, "table_path": "lazy_results.csv"},
        },
    }


def _cfg_like_pydantic(pipelines, active=None, cv=None, policy=None, enabled=True):
    # Simule PipelinesConfig Pydantic minimal avec model_dump sur les items
    def _wrap(spec):
        return SimpleNamespace(model_dump=lambda s=spec: s)

    return SimpleNamespace(
        enabled=enabled,
        active=active or [],
        cv=cv
        or {
            "type": "grid",
            "cv_folds": 3,
            "shuffle": True,
            "random_state": 0,
            "n_jobs": -1,
            "verbose": 0,
        },
        policy=policy
        or {
            "numeric": {"scaler": "StandardScaler"},
            "categorical": {"encoder": "OneHotEncoder", "handle_unknown": "ignore"},
        },
        pipelines=[_wrap(p) for p in pipelines],
    )


def _make_numeric_xy(n=100, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(rng.integers(0, 2, size=n), name="y")
    return X, y


def test_run_single_pipeline_grid(tmp_path):
    X, y = _make_numeric_xy()
    cfg = _cfg_like_pydantic(pipelines=[_svc_spec("svc_grid")], active=["svc_grid"])
    orch = PipelineOrchestrator(
        cfg=cfg, project_dir=str(tmp_path), random_state=0, ctx={"project_dir": str(tmp_path)}
    )
    orch.attach_messages(DummyMsg())
    out = orch.run(X, y)
    assert "results" in out and len(out["results"]) == 1
    assert out["results"][0]["name"] == "svc_grid"
    assert out["results"][0]["best_score"] is not None


def test_respects_active_list_and_enabled(tmp_path):
    X, y = _make_numeric_xy()
    # Deux specs, un actif par liste 'active', l'autre désactivé
    spec_a = _svc_spec("svc_a")
    spec_b = _svc_spec("svc_b")
    spec_b["enabled"] = False
    cfg = _cfg_like_pydantic(pipelines=[spec_a, spec_b], active=["svc_a"])
    orch = PipelineOrchestrator(
        cfg=cfg, project_dir=str(tmp_path), random_state=0, ctx={"project_dir": str(tmp_path)}
    )
    orch.attach_messages(DummyMsg())
    out = orch.run(X, y)
    names = [r["name"] for r in out["results"]]
    assert names == ["svc_a"]


def test_disable_column_transformer(tmp_path):
    # Données purement numériques pour tester sans ColumnTransformer
    X, y = _make_numeric_xy()
    cfg = _cfg_like_pydantic(
        pipelines=[_svc_spec("svc_no_ct", ct_enabled=False)], active=["svc_no_ct"]
    )
    orch = PipelineOrchestrator(
        cfg=cfg, project_dir=str(tmp_path), random_state=0, ctx={"project_dir": str(tmp_path)}
    )
    orch.attach_messages(DummyMsg())
    out = orch.run(X, y)
    assert out["results"][0]["name"] == "svc_no_ct"
    assert out["results"][0]["best_score"] is not None


def test_random_and_halving_random(tmp_path):
    X, y = _make_numeric_xy()
    spec = _random_spec()
    # RandomizedSearchCV
    cfg_random = _cfg_like_pydantic(
        pipelines=[spec],
        active=["svc_random"],
        cv={"type": "random", "n_iter": 8, "cv_folds": 3, "verbose": 0},
    )
    orch = PipelineOrchestrator(
        cfg=cfg_random,
        project_dir=str(tmp_path),
        random_state=0,
        ctx={"project_dir": str(tmp_path)},
    )
    orch.attach_messages(DummyMsg())
    out_rnd = orch.run(X, y)
    assert out_rnd["results"][0]["best_score"] is not None

    # HalvingRandomSearchCV
    cfg_halv = _cfg_like_pydantic(
        pipelines=[spec],
        active=["svc_random"],
        cv={"type": "halving_random", "n_iter": 8, "factor": 2, "cv_folds": 3, "verbose": 0},
    )
    orch = PipelineOrchestrator(
        cfg=cfg_halv, project_dir=str(tmp_path), random_state=0, ctx={"project_dir": str(tmp_path)}
    )
    orch.attach_messages(DummyMsg())
    out_halv = orch.run(X, y)
    assert out_halv["results"][0]["best_score"] is not None


@pytest.mark.skipif(
    pytest.importorskip("tpot", reason="TPOT non installé") is None, reason="TPOT non installé"
)
def test_automl_tpot_branch(tmp_path):
    # Test unitaire direct évaluer TPOT via l'orchestrateur (pipeline AutoML distinct)
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=120, n_features=8, n_informative=5, random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y, name="y")
    tpot_spec = {
        "name": "tpot_auto",
        "enabled": True,
        "automl": {
            "library": "tpot",
            "name": "tpot_default",
            "tpot": {
                "generations": 1,
                "population_size": 10,
                "scoring": "f1",
                "cv": 3,
                "n_jobs": -1,
                "export_best_pipeline": False,
            },
        },
    }
    cfg = _cfg_like_pydantic(
        pipelines=[tpot_spec],
        active=["tpot_auto"],
        cv={"type": "grid", "cv_folds": 3, "verbose": 0},
    )
    orch = PipelineOrchestrator(
        cfg=cfg, project_dir=str(tmp_path), random_state=0, ctx={"project_dir": str(tmp_path)}
    )
    orch.attach_messages(DummyMsg())
    out = orch.run(X, y)
    assert out["results"][0]["name"] == "tpot_auto"
    assert out["results"][0]["best_score"] is not None


@pytest.mark.skipif(
    pytest.importorskip("lazypredict", reason="lazypredict non installé") is None,
    reason="lazypredict non installé",
)
def test_automl_lazy_branch(tmp_path):
    X, y = _make_numeric_xy(n=120, p=8, seed=1)
    cfg = _cfg_like_pydantic(
        pipelines=[_lazy_spec()],
        active=["lazy_cls"],
        cv={"type": "grid", "cv_folds": 3, "verbose": 0},
    )
    orch = PipelineOrchestrator(
        cfg=cfg, project_dir=str(tmp_path), random_state=0, ctx={"project_dir": str(tmp_path)}
    )
    orch.attach_messages(DummyMsg())
    out = orch.run(X, y)
    assert out["results"][0]["name"] == "lazy_cls"
    assert out["results"][0]["cv_results_path"]
