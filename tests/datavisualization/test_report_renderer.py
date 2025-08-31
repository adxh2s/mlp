from __future__ import annotations

from src.datavisualization.report_renderer import ReportRenderer


def test_report_renderer_html_md(tmp_path, jinja_templates_dir):
    rend = ReportRenderer(str(jinja_templates_dir))
    eda_payload = {"summary_data": {"shape": {"n_samples": 10, "n_features": 3}}, "flags": {}, "profile_path": "/tmp/profile.html"}
    pipe_payload = {"results": [{"name": "svc", "best_score": 0.9, "best_params": {"C": 1}, "elapsed_sec": 0.1}]}
    out = rend.render(str(tmp_path), "demo_project", ["html", "md"], eda_payload, pipe_payload)
    assert "artifacts" in out and len(out["artifacts"]) == 2
