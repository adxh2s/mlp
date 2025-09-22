import pandas as pd
from sklearn.datasets import make_classification

from src.modeling.pipelines.evaluator import PipelineEvaluator


def small_xy(n=120, p=6, random_state=0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=4, n_classes=2, random_state=random_state
    )
    return pd.DataFrame(X), pd.Series(y, name="y")


BASE_SPEC = {
    "name": "svc_grid",
    "steps": {
        "preprocess": {
            "enabled": True,
            "imputer": "simple",
            "column_transformer": {"enabled": True, "policy": "auto"},
        },
        "estimator": {"type": "sklearn.svm.SVC", "params": {"kernel": ["linear"], "C": [0.1, 1.0]}},
    },
}

CV_GRID = {
    "type": "grid",
    "cv_folds": 3,
    "shuffle": True,
    "random_state": 0,
    "n_jobs": -1,
    "verbose": 0,
}


def test_evaluator_grid_search_exports_csv(tmp_path):
    X, y = small_xy()
    ev = PipelineEvaluator(out_dir=str(tmp_path), random_state=0)
    res = ev.evaluate(X, y, BASE_SPEC, CV_GRID, global_policy={})
    assert res["name"] == "svc_grid"
    assert res["best_score"] is not None
    assert res["cv_results_path"] and tmp_path.joinpath(res["cv_results_path"]).exists()


def test_evaluator_random_and_halving(tmp_path):
    X, y = small_xy()
    spec = {
        "name": "svc_random",
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
    for cv_type in ("random", "halving_random"):
        cv = {"type": cv_type, "n_iter": 10, "cv_folds": 3, "random_state": 0, "verbose": 0}
        ev = PipelineEvaluator(out_dir=str(tmp_path), random_state=0)
        res = ev.evaluate(X, y, spec, cv, global_policy={})
        assert res["best_score"] is not None


def test_evaluator_halving_grid(tmp_path):
    X, y = small_xy()
    spec = {
        "name": "svc_halving_grid",
        "steps": {
            "preprocess": {
                "enabled": True,
                "imputer": "simple",
                "column_transformer": {"enabled": True, "policy": "auto"},
            },
            "estimator": {
                "type": "sklearn.svm.SVC",
                "params": {"kernel": ["linear"], "C": [0.1, 1.0]},
            },
        },
    }
    cv = {"type": "halving_grid", "factor": 2, "cv_folds": 3, "verbose": 0}
    ev = PipelineEvaluator(out_dir=str(tmp_path), random_state=0)
    res = ev.evaluate(X, y, spec, cv, global_policy={})
    assert res["best_score"] is not None


def test_evaluator_tpot_branch(tmp_path):
    __import__("pytest").importorskip("tpot")
    X, y = small_xy(n=120)
    spec = {
        "name": "tpot_auto",
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
    ev = PipelineEvaluator(out_dir=str(tmp_path), random_state=0)
    res = ev._maybe_run_tpot(spec, X, y)
    assert res and res["name"] == "tpot_default"


def test_evaluator_lazy_branch(tmp_path):
    __import__("pytest").importorskip("lazypredict")
    X, y = small_xy(n=120)
    spec = {
        "name": "lazy_cls",
        "automl": {
            "library": "lazypredict",
            "name": "lazy_default",
            "lazy": {"test_size": 0.2, "top_n": 5},
        },
    }
    ev = PipelineEvaluator(out_dir=str(tmp_path), random_state=0)
    res = ev._maybe_run_lazy(spec, X, y)
    assert res and res["name"] == "lazy_default" and res["cv_results_path"]
