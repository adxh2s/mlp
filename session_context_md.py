"""
Générateur de contexte de session pour projets ML/DS.

Ce module parcourt la base de code afin de produire un fichier Markdown
(session-context.md) décrivant:
- l’arborescence des fichiers Python par dossier,
- les dépendances extraites de pyproject.toml,
- un flux des dépendances d’import,
- un résumé des docstrings,
- une section Configuration (index des YAML sous conf/),
- une section finale d’API complète (signatures + docstrings) pour les modules sous src/.

Conformité:
- PEP 8 (style), Ruff strict (qualité), annotations de type modernes.
- Imports ordonnés: standard lib → third‑party → local.
"""

from __future__ import annotations

# =========================
# Imports standard library
# =========================
import ast
import logging
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

# =========================
# Imports third‑party (optionnels)
# =========================

try:
    import yaml  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    yaml = None  # type: ignore[assignment]

# =========================
# Constantes de configuration
# =========================

DEFAULT_FILENAME: str = "session-context.md"
TEMPLATE_VERSION: str = "2.1"
SUPPORTED_PYTHON_EXTS: set[str] = {".py"}
SUPPORTED_CONFIG_EXTS: set[str] = {".yaml", ".yml", ".toml", ".json"}
MAX_DOCSTRING_LENGTH: int = 500
CLASS_DOC_PREVIEW: int = 200
METHOD_DOC_PREVIEW: int = 160
FUNCTION_DOC_PREVIEW: int = 160
LOGGER_NAME: str = "mlp.context_generator"


# =========================
# Types structurés
# =========================

class DependenciesInfo(TypedDict, total=False):
    dependencies: list[str]
    dev_dependencies: dict[str, list[str]]
    tool_config: dict[str, Any]
    build_system: dict[str, Any]
    error: str


class ApiMethodInfo(TypedDict, total=False):
    signature: str
    doc: str


class ApiClassInfo(TypedDict, total=False):
    doc: str
    methods: dict[str, ApiMethodInfo]


class ApiModuleInfo(TypedDict, total=False):
    module_doc: str
    classes: dict[str, ApiClassInfo]
    functions: dict[str, ApiMethodInfo]


class ConfigIndexEntry(TypedDict, total=False):
    keys: list[str]
    group: str
    error: str


class AdvancedSessionContextGenerator:
    """
    Génère un contexte de session complet en analysant automatiquement le projet.

    Fonctions clés:
    - Scan des fichiers Python et de configuration.
    - Extraction des dépendances depuis pyproject.toml.
    - Extraction des docstrings (module, classes, fonctions/méthodes).
    - Construction d’un graphe d’imports simplifié.
    - Index de configuration (conf/**.yaml).
    - Référence d’API exhaustive sous src/ (signatures + docstrings).
    - Rendu d’un Markdown synthétique et exploitable pour une session chat.
    """

    def __init__(self, project_name: str = "MLP", root_path: str | Path | None = None) -> None:
        """
        Initialise le générateur de contexte avancé.

        Paramètres:
        - project_name: Nom logique du projet à afficher dans le Markdown.
        - root_path: Racine du projet à analyser (par défaut le répertoire courant).
        """
        self.project_name: str = project_name
        self.root_path: Path = Path(root_path or ".").resolve()

        # Caches d’analyse
        self.project_structure: dict[str, Any] = {}
        self.dependencies: DependenciesInfo = {}
        self.python_files: list[Path] = []
        self.config_files: list[Path] = []
        self.docstrings: dict[str, dict[str, str]] = {}
        self.call_graph: dict[str, list[str]] = defaultdict(list)

        # Nouveaux index
        self.api_index: dict[str, ApiModuleInfo] = {}
        self.config_index: dict[str, ConfigIndexEntry] = {}

        # Journalisation
        self.logger = logging.getLogger(LOGGER_NAME)

    # =========================
    # Orchestration de l’analyse
    # =========================

    def analyze_project(self) -> None:
        """
        Exécute la chaîne complète d’analyse du projet.

        Étapes:
        1) Chargement optionnel de la structure projet (project_structure.yaml).
        2) Scan des fichiers Python et de configuration, avec filtrage des patterns ignorés.
        3) Parsing des dépendances (pyproject.toml).
        4) Extraction des docstrings (module/classes/fonctions).
        5) Index de l’API (signatures + docstrings) sous src/.
        6) Index des fichiers YAML sous conf/.
        7) Graphe simplifié des imports (flux d’architecture).
        """
        self._load_project_structure()
        self._scan_files()
        self._parse_dependencies()
        self._extract_docstrings()
        self._index_api()
        self._parse_yaml_configs()
        self._build_call_graph()

    # =========================
    # Collecte et parsing
    # =========================

    def _load_project_structure(self) -> None:
        """
        Charge project_structure.yaml si disponible, pour contextualiser l’arborescence.

        L’absence de PyYAML n’est pas bloquante; l’étape est simplement ignorée.
        """
        if yaml is None:
            self.logger.warning("PyYAML non disponible, chargement de structure ignoré")
            return

        structure_file = self.root_path / "project_structure.yaml"
        if structure_file.exists():
            try:
                loaded = yaml.safe_load(structure_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.project_structure = loaded
                else:
                    self.project_structure = {}
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Échec de chargement de project_structure.yaml: %s", exc)

    def _scan_files(self) -> None:
        """
        Scanne et catégorise les fichiers Python et de configuration.

        - Inclut tous les *.py.
        - Indexe les fichiers de configuration supportés (*.yaml, *.yml, *.toml, *.json).
        - Exclut les répertoires communs à ignorer (.venv, __pycache__, .git, etc.).
        """
        all_py = list(self.root_path.rglob("*.py"))

        cfgs: list[Path] = []
        for ext in SUPPORTED_CONFIG_EXTS:
            cfgs.extend(self.root_path.rglob(f"*{ext}"))

        ignore_tokens = (".venv", "__pycache__", ".git", ".pytest_cache", "node_modules")
        self.python_files = [p for p in all_py if not any(tok in str(p) for tok in ignore_tokens)]
        self.config_files = [c for c in cfgs if not any(tok in str(c) for tok in ignore_tokens)]

    def _parse_dependencies(self) -> None:
        """
        Extrait les dépendances et configurations d’outillage depuis pyproject.toml.

        Structure mise en cache:
        {
            "dependencies": [...],
            "dev_dependencies": {...},
            "tool_config": {...},
            "build_system": {...}
        }
        """
        pyproject_file = self.root_path / "pyproject.toml"
        if not pyproject_file.exists():
            self.dependencies = {}
            return

        try:
            data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Échec de parsing pyproject.toml: %s", exc)
            self.dependencies = {"error": f"Failed to parse pyproject.toml: {exc}"}
            return

        if not isinstance(data, dict):
            self.dependencies = {}
            return

        project_section = data.get("project")
        project_section = project_section if isinstance(project_section, dict) else {}

        dependencies_list_raw = project_section.get("dependencies")
        if isinstance(dependencies_list_raw, list):
            dependencies_list = [str(x) for x in dependencies_list_raw]
        else:
            dependencies_list = []

        opt_deps_raw = project_section.get("optional-dependencies")
        dev_deps: dict[str, list[str]] = {}
        if isinstance(opt_deps_raw, dict):
            for k, v in opt_deps_raw.items():
                if isinstance(v, list):
                    dev_deps[str(k)] = [str(x) for x in v]

        tool_config_raw = data.get("tool")
        tool_config = tool_config_raw if isinstance(tool_config_raw, dict) else {}

        build_system_raw = data.get("build-system")
        build_system = build_system_raw if isinstance(build_system_raw, dict) else {}

        self.dependencies = {
            "dependencies": dependencies_list,
            "dev_dependencies": dev_deps,
            "tool_config": tool_config,
            "build_system": build_system,
        }

    def _extract_docstrings(self) -> None:
        """
        Extrait les docstrings module/classes/fonctions depuis tous les fichiers Python.
        """
        for py_file in self.python_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
                rel_key = str(py_file.relative_to(self.root_path))
                bucket: dict[str, str] = {}

                mod_doc = ast.get_docstring(tree)
                if mod_doc:
                    bucket["module"] = self._truncate_docstring(mod_doc)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        cdoc = ast.get_docstring(node)
                        if cdoc:
                            bucket[f"class:{node.name}"] = self._truncate_docstring(cdoc)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fdoc = ast.get_docstring(node)
                        if fdoc:
                            bucket[f"function:{node.name}"] = self._truncate_docstring(fdoc)

                self.docstrings[rel_key] = bucket
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Échec d’extraction des docstrings pour %s: %s", py_file, exc)
                self.docstrings[str(py_file)] = {"error": str(exc)}

    def _build_call_graph(self) -> None:
        """
        Construit un graphe simplifié d’imports par fichier (liste des cibles importées).
        """
        for py_file in self.python_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
                rel_key = str(py_file.relative_to(self.root_path))

                imports: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")

                self.call_graph[rel_key] = imports
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Échec d’analyse des imports pour %s: %s", py_file, exc)

    # =========================
    # Index Configuration (conf/)
    # =========================

    def _parse_yaml_configs(self) -> None:
        """
        Indexe les YAML sous conf/ en listant les clés de premier niveau et le groupe (dossier).
        """
        if yaml is None:
            self.logger.warning("PyYAML non disponible, indexation YAML ignorée")
            return

        for cfg in self.config_files:
            rel = str(cfg.relative_to(self.root_path))
            if not (rel.startswith("conf/") and cfg.suffix in {".yaml", ".yml"}):
                continue
            try:
                loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    keys = [str(k) for k in loaded.keys()]
                else:
                    keys = []
                group = str(Path(rel).parent)
                self.config_index[rel] = {"keys": keys, "group": group}
            except Exception as exc:  # noqa: BLE001
                self.config_index[rel] = {"error": str(exc), "group": str(Path(rel).parent)}

    def _generate_configuration_section(self) -> str:
        """
        Génère la section Markdown 'Configuration' groupée par sous‑dossiers de conf/.
        """
        lines: list[str] = ["### ⚙️ Configuration"]
        if not self.config_index:
            lines.append("Aucun fichier de configuration YAML indexé ou PyYAML manquant.")
            return "\n".join(lines)

        by_group: dict[str, list[str]] = defaultdict(list)
        for rel, meta in self.config_index.items():
            group = meta.get("group") or "conf"
            by_group[group].append(rel)

        for group in sorted(by_group):
            lines.append(f"**{group}:**")
            for rel in sorted(by_group[group]):
                meta = self.config_index.get(rel, {})
                if "error" in meta:
                    lines.append(f"- `{rel}` — erreur: {meta['error']}")
                else:
                    keys = meta.get("keys", [])
                    kstr = ", ".join(keys) if keys else "∅"
                    lines.append(f"- `{rel}` — clés: {kstr}")
            lines.append("")

        lines.append("_Cette configuration sera détaillée par les orchestrateurs correspondants._")
        return "\n".join(lines)

    # =========================
    # Index API (src/) avec signatures
    # =========================

    def _index_api(self) -> None:
        """
        Construit un index d’API pour les modules sous src/ incluant signatures et docstrings.
        """
        for py_file in self.python_files:
            rel = str(py_file.relative_to(self.root_path))
            if not rel.startswith("src/"):
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)

                module_doc = ast.get_docstring(tree) or ""
                classes: dict[str, ApiClassInfo] = {}
                functions: dict[str, ApiMethodInfo] = {}

                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        cdoc = ast.get_docstring(node) or ""
                        methods: dict[str, ApiMethodInfo] = {}
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                mdoc = ast.get_docstring(item) or ""
                                methods[item.name] = {
                                    "signature": self._format_signature(item),
                                    "doc": mdoc,
                                }
                        classes[node.name] = {"doc": cdoc, "methods": methods}
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fdoc = ast.get_docstring(node) or ""
                        functions[node.name] = {
                            "signature": self._format_signature(node),
                            "doc": fdoc,
                        }

                self.api_index[rel] = {
                    "module_doc": module_doc,
                    "classes": classes,
                    "functions": functions,
                }
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Index API échoué pour %s: %s", rel, exc)

    # --- Helpers de formatage de signature (réduction de complexité) ---

    @staticmethod
    def _ann_str(a: ast.AST | None) -> str:
        if a is None:
            return ""
        try:
            return ast.unparse(a)
        except Exception:  # noqa: BLE001
            return "..."

    @staticmethod
    def _default_str(d: ast.AST | None) -> str:
        if d is None:
            return ""
        try:
            return ast.unparse(d)
        except Exception:  # noqa: BLE001
            return "..."

    def _format_posonly(self, args: ast.arguments) -> list[str]:
        parts: list[str] = []
        for a in getattr(args, "posonlyargs", []):
            ann = f": {self._ann_str(a.annotation)}" if a.annotation else ""
            parts.append(f"{a.arg}{ann}")
        return parts

    def _format_poskw(self, args: ast.arguments) -> list[str]:
        parts: list[str] = []
        for a in args.args:
            ann = f": {self._ann_str(a.annotation)}" if a.annotation else ""
            parts.append(f"{a.arg}{ann}")
        # Valeurs par défaut alignées sur la fin
        if args.defaults:
            start = len(parts) - len(args.defaults)
            for i, d in enumerate(args.defaults, start=start):
                parts[i] += f" = {self._default_str(d)}"
        return parts

    def _format_vararg(self, args: ast.arguments) -> list[str]:
        if not args.vararg:
            return []
        ann = f": {self._ann_str(args.vararg.annotation)}" if args.vararg.annotation else ""
        return [f"*{args.vararg.arg}{ann}"]

    def _format_kwonly(self, args: ast.arguments) -> list[str]:
        parts: list[str] = []
        for i, a in enumerate(args.kwonlyargs):
            ann = f": {self._ann_str(a.annotation)}" if a.annotation else ""
            comp = f"{a.arg}{ann}"
            if args.kw_defaults and args.kw_defaults[i] is not None:
                comp += f" = {self._default_str(args.kw_defaults[i])}"
            parts.append(comp)
        return parts

    def _format_kwargs(self, args: ast.arguments) -> list[str]:
        if not args.kwarg:
            return []
        ann = f": {self._ann_str(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        return [f"**{args.kwarg.arg}{ann}"]

    def _format_signature(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """
        Reconstruit une signature Python‑like depuis un nœud AST FunctionDef/AsyncFunctionDef.
        """
        args = func.args

        posonly = self._format_posonly(args)
        poskw = self._format_poskw(args)
        star = self._format_vararg(args)
        kwonly = self._format_kwonly(args)
        dstar = self._format_kwargs(args)

        params: list[str] = []
        if posonly:
            params.extend(posonly)
            params.append("/")
        params.extend(poskw)
        if star or kwonly:
            if not star and kwonly:
                params.append("*")
            params.extend(star)
            params.extend(kwonly)
        params.extend(dstar)

        returns = self._ann_str(getattr(func, "returns", None))
        sig = f"({', '.join(params)})"
        if returns:
            sig += f" -> {returns}"
        return sig

    # =========================
    # Rendu Markdown
    # =========================

    def _truncate_docstring(self, docstring: str) -> str:
        """
        Tronque une docstring au‑delà de MAX_DOCSTRING_LENGTH pour lisibilité.
        """
        if len(docstring) <= MAX_DOCSTRING_LENGTH:
            return docstring
        return f"{docstring[:MAX_DOCSTRING_LENGTH]}..."

    def _generate_file_listing(self) -> str:
        """
        Produit une liste des fichiers Python regroupés par dossier, avec aperçu de docstring.
        """
        sections: list[str] = ["### 📁 Python Files by Directory"]
        py_by_dir: dict[str, list[Path]] = defaultdict(list)
        for py_file in self.python_files:
            parent = py_file.parent.relative_to(self.root_path)
            py_by_dir[str(parent)].append(py_file)

        for directory, files in sorted(py_by_dir.items()):
            sections.append(f"**{directory}/:**")
            for file in sorted(files):
                rel_path = file.relative_to(self.root_path)
                doc_preview = ""
                docs = self.docstrings.get(str(rel_path), {})
                if "module" in docs:
                    doc_preview = f" - {docs['module'][:100]}..."
                sections.append(f"- `{file.name}`{doc_preview}")
            sections.append("")
        return "\n".join(sections)

    def _generate_dependency_analysis(self) -> str:
        """
        Génère la section des dépendances à partir de self.dependencies.
        """
        if not self.dependencies:
            return "No dependency information found."

        sections: list[str] = []
        deps = self.dependencies.get("dependencies", [])
        if deps:
            sections.append("### 📦 Production Dependencies")
            for dep in deps:
                sections.append(f"- `{dep}`")
            sections.append("")

        dev = self.dependencies.get("dev_dependencies", {})
        if dev:
            sections.append("### 🔧 Development Dependencies")
            for category, dlist in dev.items():
                sections.append(f"**{category}:**")
                for dep in dlist:
                    sections.append(f"- `{dep}`")
            sections.append("")

        tools = self.dependencies.get("tool_config", {})
        if tools:
            sections.append("### ⚙️ Configured Tools")
            tool_names = [str(name) for name in tools.keys()]
            sections.append(", ".join(f"`{name}`" for name in tool_names))
            sections.append("")

        return "\n".join(sections)

    def _generate_architecture_flow(self) -> str:
        """
        Génère une vue sommaire du flux d’imports (points d’entrée, orchestrateurs).
        """
        sections: list[str] = ["### 🔄 Import Dependencies Flow"]

        main_files = [f for f in self.call_graph.keys() if f.endswith("main.py")]
        if main_files:
            sections.append("**Main Entry Points:**")
            for mf in main_files:
                sections.append(f"- `{mf}`")
                for imp in self.call_graph.get(mf, [])[:5]:
                    sections.append(f" └── {imp}")
            sections.append("")

        orch_files = [f for f in self.call_graph.keys() if "orchestrator" in f]
        if orch_files:
            sections.append("**Orchestrators Chain:**")
            for of in orch_files:
                sections.append(f"- `{of}`")

        return "\n".join(sections)

    def _generate_docstring_summary(self) -> str:
        """
        Résume les docstrings de module pour les fichiers couverts.
        """
        sections: list[str] = ["### 📚 Key Components Documentation"]
        for file_path, docs in self.docstrings.items():
            mod = docs.get("module")
            if mod:
                sections.append(f"**{file_path}:**")
                sections.append(mod)
                sections.append("")
        return "\n".join(sections)

    def _generate_api_reference_section(self) -> str:
        """
        Rend la section finale 'API complète (src)' avec signatures et docstrings.
        """
        lines: list[str] = ["### 🧭 API complète (src)"]
        if not self.api_index:
            lines.append("Aucune API indexée sous src/.")
            return "\n".join(lines)

        for module in sorted(self.api_index.keys()):
            entry = self.api_index[module]
            lines.append(f"**{module}**")
            mdoc = str(entry.get("module_doc", "") or "").strip()
            if mdoc:
                lines.append(mdoc)

            classes = entry.get("classes", {}) or {}
            if classes:
                lines.append("Classes:")
                for cname in sorted(classes.keys()):
                    c = classes[cname]
                    cdoc = str(c.get("doc", "") or "").strip()
                    short = f"{cdoc[:CLASS_DOC_PREVIEW]}..." if len(cdoc) > CLASS_DOC_PREVIEW else cdoc
                    lines.append(f"- {cname}: {short}")
                    methods = c.get("methods", {}) or {}
                    for mname in sorted(methods.keys()):
                        m = methods[mname]
                        sig = m.get("signature", "()")
                        mdoc = str(m.get("doc", "") or "").strip()
                        mshort = (
                            f"{mdoc[:METHOD_DOC_PREVIEW]}..." if len(mdoc) > METHOD_DOC_PREVIEW else mdoc
                        )
                        lines.append(f"  - {mname}{sig}: {mshort}")

            funcs = entry.get("functions", {}) or {}
            if funcs:
                lines.append("Fonctions:")
                for fname in sorted(funcs.keys()):
                    f = funcs[fname]
                    sig = f.get("signature", "()")
                    fdoc = str(f.get("doc", "") or "").strip()
                    fshort = (
                        f"{fdoc[:FUNCTION_DOC_PREVIEW]}..."
                        if len(fdoc) > FUNCTION_DOC_PREVIEW
                        else fdoc
                    )
                    lines.append(f"- {fname}{sig}: {fshort}")

            lines.append("")
        return "\n".join(lines)

    # =========================
    # Gabarit final
    # =========================

    def _build_comprehensive_template(self) -> str:
        """
        Construit le Markdown complet à partir des données collectées.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        overview = (
            f"# {self.project_name} Project Session Context\n\n"
            "## 📊 Project Overview\n"
            "- **Architecture**: Automated analysis of modular ML pipeline\n"
            f"- **Last Updated**: {timestamp}\n"
            f"- **Files Analyzed**: {len(self.python_files)} Python files, {len(self.config_files)} config files\n"
            f"- **Root Path**: `{self.root_path}`\n"
        )

        results = (
            "## 🎯 Automated Analysis Results\n\n"
            f"{self._generate_file_listing()}\n\n"
            f"{self._generate_dependency_analysis()}\n\n"
            f"{self._generate_architecture_flow()}\n\n"
            f"{self._generate_docstring_summary()}\n\n"
            f"{self._generate_configuration_section()}\n\n"
        )

        standards = (
            "## 🔧 Coding Standards (Auto-Applied)\n"
            "- **Style**: PEP8 + Ruff strict compliance\n"
            "- **Docstrings**: Mandatory (summary + detailed description)\n"
            "- **Imports**: Ordered (stdlib → third-party → local)\n"
            "- **Constants**: UPPERCASE at class top\n"
            "- **Types**: PEP 604 unions (X | None), builtin generics (dict/list/tuple)\n"
            "- **Format**: Production-ready, copy-pastable code blocks\n\n"
        )

        structure = (
            "## 📁 Project Structure Analysis\n"
            f"{self.project_name}/\n"
            f"├── Python files: {len(self.python_files)}\n"
            f"├── Config files: {len(self.config_files)}\n"
            f"├── Total modules with docs: {len([f for f in self.docstrings.values() if 'module' in f])}\n"
            f"└── Dependencies analyzed: {len(self.dependencies.get('dependencies', []))}\n\n"
        )

        current_state = (
            "## 🚀 Current State (Auto-Detected)\n"
            "- ✅ Files discovered and analyzed automatically\n"
            "- ✅ Dependencies extracted from configuration\n"
            "- ✅ Docstrings catalogued and summarized\n"
            "- ✅ Import relationships mapped\n"
            "- ✅ Architecture flow documented\n\n"
        )

        usage = (
            "## 📋 Session Usage\n"
            f"1. Start new chat with: \"Context: {self.project_name} project from session-context.md\"\n"
            "2. Attach this file to provide immediate context\n"
            "3. Mention specific files from the analysis above\n"
            "4. Standards auto-applied - no need to re-specify\n\n"
        )

        insights = (
            "## 🎯 Analysis Insights\n"
            f"- **Most documented module**: {self._find_most_documented_module()}\n"
            f"- **Main entry points**: {len([f for f in self.call_graph.keys() if 'main' in f])} detected\n"
            f"- **Orchestrator pattern**: "
            f"{'✅ Detected' if any('orchestrator' in str(f) for f in self.python_files) else '❌ Not found'}\n\n"
        )

        api = f"{self._generate_api_reference_section()}\n\n"

        footer = (
            "---\n"
            f"*Generated by AdvancedSessionContextGenerator v{TEMPLATE_VERSION}*\n"
            f"*Automated analysis of {len(self.python_files)} files completed at {timestamp}*\n"
        )

        parts = [overview, results, standards, structure, current_state, usage, insights, api, footer]
        return "\n".join(parts)

    def _find_most_documented_module(self) -> str:
        """
        Retourne le module avec le plus de docstrings collectés (heuristique simple).
        """
        max_docs = 0
        best_module = "None"
        for file_path, docs in self.docstrings.items():
            doc_count = len(docs)
            if doc_count > max_docs:
                max_docs = doc_count
                best_module = file_path
        return best_module if max_docs > 1 else "None found"

    # =========================
    # Sortie fichier
    # =========================

    def generate_context_md(self, output_path: str | Path | None = None) -> Path:
        """
        Lance l’analyse et écrit le fichier Markdown final.

        Paramètres:
        - output_path: Chemin du fichier de sortie; DEFAULT_FILENAME si None.

        Retour:
        - Chemin absolu du fichier généré.
        """
        self.analyze_project()
        out = Path(output_path or DEFAULT_FILENAME)
        content = self._build_comprehensive_template()
        out.write_text(content, encoding="utf-8")
        self.logger.info("Session context généré: %s", out)
        return out.resolve()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generator = AdvancedSessionContextGenerator("MLP", ".")
    path = generator.generate_context_md()
    print(f"✅ Contexte intelligent généré: {path}")
    print(f"📊 Analysé: {len(generator.python_files)} fichiers Python")
    print(f"📦 Dépendances: {len(generator.dependencies.get('dependencies', []))}")
    print("📎 Attachez ce fichier à vos nouveaux fils de conversation")
