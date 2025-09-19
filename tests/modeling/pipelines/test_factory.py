import numpy as np
import pandas as pd
from src.modeling.pipelines.factory import PipelineFactory

def make_fake_df(n=64, p_num=3, p_cat=2):
    X = pd.DataFrame(
        np.c_[np.random.randn(n, p_num), np.random.choice(list("ABC"), size=(n, p_cat))],
        columns=[*(f"num{i}" for i in range(p_num)), *(f"cat{i}" for i in range(p_cat))],
    )
    y = pd.Series(np.random.randint(0, 2, size=n), name="y")
    return X, y

def test_factory_auto_policy_builds_ct_and_grids():
    X, _ = make_fake_df()
    spec = {
        "name": "auto_svc",
        "steps": {
            "preprocess": {
                "enabled": True,
                "imputer": "simple",
                "column_transformer": {"enabled": True, "policy": "auto"},
            },
            "estimator": {"type": "sklearn.svm.SVC", "params": {"C": [0.1, 1.0]}},
        },
    }
    global_policy = {
        "numeric": {"scaler": "StandardScaler"},
        "categorical": {"encoder": "OneHotEncoder", "handle_unknown": "ignore"},
    }
    pipe, grid, dists = PipelineFactory.build(spec, global_policy)
    assert "ct" in dict(pipe.steps) and "estimator" in dict(pipe.steps)
    assert "estimator__C" in grid and grid["estimator__C"] == [0.1, 1.0]
    assert dists == {}
