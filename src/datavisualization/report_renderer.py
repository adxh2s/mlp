from __future__ import annotations

"""ReportRenderer: render HTML/Markdown report from templates and a data context."""

# Décorateurs: import robuste avec fallback no-op
try:
    from src.instrumentation.decorators import log_call
except Exception:  # pragma: no cover
    from typing import Callable, TypeVar, ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

    def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:  # type: ignore[override]
        def deco(fn: Callable[P, T]) -> Callable[P, T]:
            return fn
        return deco

import os
import time
import uuid
from typing import Any

import jinja2


class ReportRenderer:
    """Render HTML/Markdown report from templates and a data context."""

    KEY_REPORT_ID = "report_id"
    KEY_ARTIFACTS = "artifacts"

    HTML_EXT = ".html"
    MD_EXT = ".md"

    @log_call("report_renderer.__init__")
    def __init__(self, templates_dir: str) -> None:
        """Initialize the renderer with a filesystem templates directory."""
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            autoescape=jinja2.select_autoescape(enabled_extensions=("html",)),
        )

    @log_call("report_renderer._build_context")
    def _build_context(
        self,
        project_name: str,
        eda_payload: dict[str, Any],
        pipe_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a normalized context dict consumed by templates."""
        summary = eda_payload.get("summary_data", {}) or {}
        flags = eda_payload.get("flags", {}) or {}
        results = pipe_payload.get("results", []) or []

        return {
            "project_name": project_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eda": {
                "summary": summary,
                "flags": flags,
                "profile_path": eda_payload.get("profile_path"),
                "summary_path": eda_payload.get("summary_path"),
            },
            "pipeline": {
                "results": results,
            },
        }

    @log_call("report_renderer.render")
    def render(
        self,
        out_dir: str,
        project_name: str,
        formats: list[str],
        eda_payload: dict[str, Any],
        pipe_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Render report in requested formats and return artifact paths."""
        os.makedirs(out_dir, exist_ok=True)
        report_id = str(uuid.uuid4())[:8]
        context = self._build_context(project_name, eda_payload, pipe_payload)

        artifacts: list[str] = []

        if "html" in formats:
            template = self.env.get_template("report.html.jinja")
            html = template.render(**context)
            html_path = os.path.join(out_dir, f"report_{report_id}{self.HTML_EXT}")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            artifacts.append(html_path)

        if "md" in formats:
            template = self.env.get_template("report.md.jinja")
            md = template.render(**context)
            md_path = os.path.join(out_dir, f"report_{report_id}{self.MD_EXT}")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)
            artifacts.append(md_path)

        return {self.KEY_REPORT_ID: report_id, self.KEY_ARTIFACTS: artifacts}
