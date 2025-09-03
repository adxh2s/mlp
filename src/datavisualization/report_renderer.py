from __future__ import annotations

import os
import time
import uuid
from typing import Any

import jinja2


class ReportRenderer:
    """Render HTML/Markdown reports from templates and a data context.

    Uses Jinja2 templates loaded from a filesystem directory to separate
    presentation from logic and to support reusable layouts.
    """

    KEY_REPORT_ID = "report_id"
    KEY_ARTIFACTS = "artifacts"
    HTML_EXT = ".html"
    MD_EXT = ".md"

    def __init__(self, templates_dir: str) -> None:
        """Initialize the renderer with a filesystem templates directory.

        Args:
            templates_dir: Directory containing Jinja2 templates.
        """
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            autoescape=jinja2.select_autoescape(enabled_extensions=("html",)),
        )

    def _build_context(
        self,
        project_name: str,
        eda_payload: dict[str, Any],
        pipe_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a normalized context dict consumed by templates.

        Args:
            project_name: Project display name.
            eda_payload: EDA orchestrator output payload.
            pipe_payload: Pipelines orchestrator output payload.

        Returns:
            A dictionary with keys project_name, generated_at, eda, results.
        """
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
            },
            "results": results,
        }

    def render(
        self,
        out_dir: str,
        project_name: str,
        formats: list[str],
        eda_payload: dict[str, Any],
        pipe_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Render reports in the requested formats and return artifact paths.

        Args:
            out_dir: Output directory for rendered reports.
            project_name: Project display name.
            formats: List of formats to render ("html", "md").
            eda_payload: EDA orchestrator output payload.
            pipe_payload: Pipelines orchestrator output payload.

        Returns:
            A mapping containing a unique report_id and the artifact file paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        report_id = str(uuid.uuid4())[:8]
        ctx = self._build_context(project_name, eda_payload, pipe_payload)
        artifacts: list[str] = []

        if "html" in formats:
            template = self.env.get_template("report.html.jinja")
            html = template.render(**ctx)
            html_path = os.path.join(out_dir, f"report_{report_id}{self.HTML_EXT}")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            artifacts.append(html_path)

        if "md" in formats:
            template = self.env.get_template("report.md.jinja")
            md = template.render(**ctx)
            md_path = os.path.join(out_dir, f"report_{report_id}{self.MD_EXT}")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)
            artifacts.append(md_path)

        return {self.KEY_REPORT_ID: report_id, self.KEY_ARTIFACTS: artifacts}
