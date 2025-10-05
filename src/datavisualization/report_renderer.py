# src/datavisualization/report_renderer.py
from __future__ import annotations

"""
ReportRenderer: moteur de rendu HTML/Markdown à partir de templates Jinja2 et d'un contexte EDA + pipeline.

- API:
  - build_context(project_name, eda_payload, pipe_payload) -> dict JSON-safe
  - render(out_dir, project_name, formats, eda_payload, pipe_payload, **kwargs) -> dict {report_id, artifacts, main}
- Compatibilité:
  - Accepte aussi les alias historiques: outdir, projectname, edapayload, pipepayload.
"""

from typing import Any, Iterable, Mapping, Sequence
import os
import time
import uuid

import jinja2

# Décorateur optionnel: fallback no-op si l'instrumentation n'est pas disponible
try:
    from src.instrumentation.decorators import log_call
except Exception:  # pragma: no cover
    def log_call(name: str | None = None):
        def deco(fn):
            return fn
        return deco

# Dépendances optionnelles pour rendre le contexte JSON-safe
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# Constantes (PEP 8: UPPER_SNAKE_CASE)
KEY_PROJECT_NAME: str = "project_name"
KEY_GENERATED_AT: str = "generated_at"
KEY_EDA: str = "eda"
KEY_PIPELINE: str = "pipeline"
KEY_SUMMARY: str = "summary"
KEY_FLAGS: str = "flags"
KEY_PROFILE_PATH: str = "profile_path"
KEY_SUMMARY_PATH: str = "summary_path"
KEY_RESULTS: str = "results"

KEY_REPORT_ID: str = "report_id"
KEY_ARTIFACTS: str = "artifacts"
KEY_MAIN: str = "main"

HTML_EXT: str = ".html"
MD_EXT: str = ".md"

DEFAULT_HTML_TEMPLATE: str = "report.html.jinja"
DEFAULT_MD_TEMPLATE: str = "report.md.jinja"


def _is_iterable_but_not_str(x: Any) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    return isinstance(x, Iterable)


class ReportRenderer:
    """
    Render HTML/Markdown report from templates and a normalized data context.
    Uses Jinja2 templates loaded from a filesystem directory to separate presentation
    from logic and to support reusable layouts.
    """

    @log_call("report_renderer.__init__")
    def __init__(self, templates_dir: str) -> None:
        """
        Initialize the renderer with a filesystem templates directory.

        Args:
            templates_dir: Directory containing Jinja2 templates.
        """
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            autoescape=jinja2.select_autoescape(enabled_extensions=("html",)),
        )

    # ---------- JSON-safe normalization utilities ----------

    def _json_safe_scalar(self, v: Any) -> Any:
        # numpy scalars
        if np is not None and isinstance(v, (np.generic,)):
            return v.item()
        # pandas NA
        if pd is not None:
            try:
                from pandas._libs.missing import NAType  # type: ignore
                if isinstance(v, NAType):  # pragma: no cover
                    return None
            except Exception:
                pass
        # python scalars unchanged
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        # datetime-like
        if hasattr(v, "isoformat"):
            try:
                return v.isoformat()  # type: ignore[attr-defined]
            except Exception:
                pass
        # fallback: str
        return str(v)

    def _json_safe_sequence(self, seq: Iterable[Any]) -> list[Any]:
        out: list[Any] = []
        for item in seq:
            out.append(self._json_safe_value(item))
        return out

    def _json_safe_mapping(self, mp: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in mp.items():
            out[str(k)] = self._json_safe_value(v)
        return out

    def _json_safe_value(self, v: Any) -> Any:
        # pandas objects
        if pd is not None:
            if "Series" in type(v).__name__:
                try:
                    return self._json_safe_sequence(v.tolist())  # type: ignore[attr-defined]
                except Exception:
                    pass
            if "DataFrame" in type(v).__name__:
                try:
                    return self._json_safe_mapping(v.to_dict(orient="list"))  # type: ignore[attr-defined]
                except Exception:
                    pass

        # mappings
        if isinstance(v, Mapping):
            return self._json_safe_mapping(v)  # type: ignore[return-value]
        # sequences
        if _is_iterable_but_not_str(v):
            try:
                return self._json_safe_sequence(v)  # type: ignore[arg-type]
            except Exception:
                pass
        # scalars
        return self._json_safe_scalar(v)

    # ---------- Context building ----------

    @log_call("report_renderer.build_context")
    def build_context(
        self,
        project_name: str,
        eda_payload: dict[str, Any] | None,
        pipe_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Build a normalized context dict consumed by templates.

        Args:
            project_name: Project display name.
            eda_payload: EDA orchestrator output payload.
            pipe_payload: Pipeline orchestrator output payload.

        Returns:
            A dictionary with keys project_name, generated_at, eda, pipeline.
        """
        eda_payload = eda_payload or {}
        pipe_payload = pipe_payload or {}

        summary = self._json_safe_value(eda_payload.get("summary_data") or eda_payload.get("summary") or {})
        flags = self._json_safe_value(eda_payload.get("flags") or {})
        profile_path = self._json_safe_value(eda_payload.get("profile_path"))
        summary_path = self._json_safe_value(eda_payload.get("summary_path"))

        results = self._json_safe_value(
            pipe_payload.get("results")
            or pipe_payload.get("pipelines")
            or pipe_payload.get("metrics")
            or []
        )

        ctx = {
            KEY_PROJECT_NAME: project_name,
            KEY_GENERATED_AT: time.strftime("%Y-%m-%d %H:%M:%S"),
            KEY_EDA: {
                KEY_SUMMARY: summary,
                KEY_FLAGS: flags,
                KEY_PROFILE_PATH: profile_path,
                KEY_SUMMARY_PATH: summary_path,
            },
            KEY_PIPELINE: {
                KEY_RESULTS: results,
            },
        }
        return ctx

    # ---------- Rendering ----------

    def _select_formats(self, formats: Sequence[str] | None) -> set[str]:
        if not formats:
            return {"html"}
        norm = {str(f).strip().lower() for f in formats if str(f).strip()}
        aliases = {"markdown": "md", "htm": "html"}
        normalized = {aliases.get(f, f) for f in norm}
        return {f for f in normalized if f in {"html", "md"}}

    def _render_html(self, ctx: dict[str, Any]) -> str:
        tpl = self.env.get_template(DEFAULT_HTML_TEMPLATE)
        return tpl.render(ctx)

    def _render_md(self, ctx: dict[str, Any]) -> str:
        tpl = self.env.get_template(DEFAULT_MD_TEMPLATE)
        return tpl.render(ctx)

    @log_call("report_renderer.render")
    def render(
        self,
        out_dir: str | None = None,
        project_name: str | None = None,
        formats: Sequence[str] | None = None,
        eda_payload: dict[str, Any] | None = None,
        pipe_payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Render report files to out_dir according to requested formats.

        Args:
            out_dir: Output directory to write artifacts into.
            project_name: Display name for the project.
            formats: Sequence of formats, e.g. ["html", "md"].
            eda_payload: EDA results payload.
            pipe_payload: Pipeline results payload.
            **kwargs: Accepts alias names: outdir, projectname, edapayload, pipepayload.

        Returns:
            dict with keys:
              - report_id: str UUID
              - artifacts: list of written artifact paths
              - main: path to the main artifact (HTML preferred)
        """
        # Alias mapping for backward compatibility
        if out_dir is None:
            out_dir = kwargs.get("outdir") or kwargs.get("out_dir")
        if project_name is None:
            project_name = kwargs.get("projectname") or kwargs.get("project_name") or "MLP Project"
        if eda_payload is None:
            eda_payload = kwargs.get("edapayload") or kwargs.get("eda_payload") or {}
        if pipe_payload is None:
            pipe_payload = kwargs.get("pipepayload") or kwargs.get("pipe_payload") or {}

        fmts = self._select_formats(formats)
        if not fmts:
            fmts = {"html"}

        ctx = self.build_context(project_name, eda_payload, pipe_payload)

        # When no out_dir is provided, return first rendered string as 'main' without files
        if not out_dir:
            report_id = str(uuid.uuid4())
            main_str = self._render_html(ctx) if "html" in fmts else self._render_md(ctx)
            return {
                KEY_REPORT_ID: report_id,
                KEY_ARTIFACTS: [],
                KEY_MAIN: main_str,
            }

        os.makedirs(out_dir, exist_ok=True)
        report_id = str(uuid.uuid4())
        artifacts: list[str] = []
        main_path: str | None = None

        # HTML
        if "html" in fmts:
            html = self._render_html(ctx)
            html_path = os.path.join(out_dir, f"report_{report_id}{HTML_EXT}")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            artifacts.append(html_path)
            main_path = main_path or html_path

        # Markdown
        if "md" in fmts:
            # Render only if the template exists to avoid exceptions when MD template is not provided
            try:
                md = self._render_md(ctx)
                md_path = os.path.join(out_dir, f"report_{report_id}{MD_EXT}")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md)
                artifacts.append(md_path)
                # If no HTML, consider MD as main
                main_path = main_path or md_path
            except jinja2.exceptions.TemplateNotFound:
                # Silently skip if MD template is absent
                pass

        return {
            KEY_REPORT_ID: report_id,
            KEY_ARTIFACTS: artifacts,
            KEY_MAIN: main_path or (artifacts[0] if artifacts else ""),
        }
