from __future__ import annotations

import pandas as pd

from src.datanalysis.eda_summary import EDASummary


def test_eda_summary_basic(tmp_outputs, demo_dataset):
    X, y = demo_dataset
    path, summary, flags = EDASummary.summarize(X, y, str(tmp_outputs))
    assert path.endswith(".json")
    assert summary["shape"]["n_samples"] == len(X)
    assert "duplicates" in summary
    assert isinstance(flags["needs_scaling"], bool)
    assert isinstance(flags["high_dimensional"], bool)
