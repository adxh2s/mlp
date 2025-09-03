from __future__ import annotations

# Standard library
import ast
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import tomllib

# Third-party
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


class AdvancedSessionContextGenerator:
    """Generate comprehensive session context by analyzing project structure automatically.

    Analyzes project structure, dependencies, docstrings, and code flows
    to generate intelligent context documentation for chat sessions.
    """

    # Class constants
    DEFAULT_FILENAME = "session-context.md"
    TEMPLATE_VERSION = "2.0"
    SUPPORTED_PYTHON_EXTS = {".py"}
    SUPPORTED_CONFIG_EXTS = {".yaml", ".yml", ".toml", ".json"}
    MAX_DOCSTRING_LENGTH = 500
    LOGGER_NAME = "mlp.context_generator"

    def __init__(self, project_name: str = "MLP", root_path: str | Path | None = None) -> None:
        """Initialize advanced context generator for comprehensive project analysis."""
        self.project_name = project_name
        self.root_path = Path(root_path or ".")
        self.project_structure: dict[str, Any] = {}
        self.dependencies: dict[str, Any] = {}
        self.python_files: list[Path] = []
        self.config_files: list[Path] = []
        self.docstrings: dict[str, dict[str, str]] = {}
        self.call_graph: dict[str, list[str]] = defaultdict(list)
        self.logger = logging.getLogger(self.LOGGER_NAME)

    def analyze_project(self) -> None:
        """Perform comprehensive project analysis."""
        self._load_project_structure()
        self._scan_files()
        self._parse_dependencies()
        self._extract_docstrings()
        self._build_call_graph()

    def _load_project_structure(self) -> None:
        """Load project structure from project_structure.yaml if available."""
        if yaml is None:
            self.logger.warning("PyYAML not available, skipping project structure loading")
            return

        structure_file = self.root_path / "project_structure.yaml"
        if structure_file.exists():
            try:
                with structure_file.open("r", encoding="utf-8") as f:
                    self.project_structure = yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Failed to load project structure: {e}")

    def _scan_files(self) -> None:
        """Scan and categorize all relevant files in the project."""
        # Python files
        self.python_files = list(self.root_path.rglob("*.py"))

        # Configuration files
        for ext in self.SUPPORTED_CONFIG_EXTS:
            self.config_files.extend(self.root_path.rglob(f"*{ext}"))

        # Filter out common ignore patterns
        ignore_patterns = {".venv", "__pycache__", ".git", ".pytest_cache", "node_modules"}
        self.python_files = [
            f
            for f in self.python_files
            if not any(pattern in str(f) for pattern in ignore_patterns)
        ]

    def _parse_dependencies(self) -> None:
        """Parse dependencies from pyproject.toml or similar files."""
        pyproject_file = self.root_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with pyproject_file.open("rb") as f:
                    data = tomllib.load(f)

                # Extract project dependencies
                project_deps = data.get("project", {}).get("dependencies", [])
                dev_deps = data.get("project", {}).get("optional-dependencies", {})
                tool_config = data.get("tool", {})

                self.dependencies = {
                    "dependencies": project_deps,
                    "dev_dependencies": dev_deps,
                    "tool_config": tool_config,
                    "build_system": data.get("build-system", {}),
                }
            except Exception as e:
                self.logger.error(f"Failed to parse pyproject.toml: {e}")
                self.dependencies = {"error": f"Failed to parse pyproject.toml: {e}"}

    def _extract_docstrings(self) -> None:
        """Extract module, class, and function docstrings from all Python files."""
        for py_file in self.python_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)

                file_key = str(py_file.relative_to(self.root_path))
                self.docstrings[file_key] = {}

                # Module docstring
                module_doc = ast.get_docstring(tree)
                if module_doc:
                    self.docstrings[file_key]["module"] = self._truncate_docstring(module_doc)

                # Class and function docstrings
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_doc = ast.get_docstring(node)
                        if class_doc:
                            self.docstrings[file_key][f"class:{node.name}"] = (
                                self._truncate_docstring(class_doc)
                            )
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_doc = ast.get_docstring(node)
                        if func_doc:
                            self.docstrings[file_key][f"function:{node.name}"] = (
                                self._truncate_docstring(func_doc)
                            )

            except Exception as e:
                self.logger.warning(f"Failed to extract docstrings from {py_file}: {e}")
                self.docstrings[str(py_file)] = {"error": str(e)}

    def _build_call_graph(self) -> None:
        """Build basic call graph by analyzing imports and function calls."""
        for py_file in self.python_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)

                file_key = str(py_file.relative_to(self.root_path))
                imports = []

                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                full_name = f"{node.module}.{alias.name}"
                                imports.append(full_name)

                self.call_graph[file_key] = imports

            except Exception as e:
                self.logger.warning(f"Failed to analyze imports in {py_file}: {e}")

    def _truncate_docstring(self, docstring: str) -> str:
        """Truncate long docstrings for readability."""
        if len(docstring) <= self.MAX_DOCSTRING_LENGTH:
            return docstring
        return docstring[: self.MAX_DOCSTRING_LENGTH] + "..."

    def _generate_file_listing(self) -> str:
        """Generate formatted file listing."""
        sections = []

        # Python files by directory
        py_by_dir: dict[str, list[Path]] = defaultdict(list)
        for py_file in self.python_files:
            parent = py_file.parent.relative_to(self.root_path)
            py_by_dir[str(parent)].append(py_file)

        sections.append("### 📁 Python Files by Directory")
        for directory, files in sorted(py_by_dir.items()):
            sections.append(f"**{directory}/:**")
            for file in sorted(files):
                rel_path = file.relative_to(self.root_path)
                # Add docstring preview if available
                doc_preview = ""
                if str(rel_path) in self.docstrings and "module" in self.docstrings[str(rel_path)]:
                    doc_preview = f" - {self.docstrings[str(rel_path)]['module'][:100]}..."
                sections.append(f"- `{file.name}`{doc_preview}")
            sections.append("")

        return "\n".join(sections)

    def _generate_dependency_analysis(self) -> str:
        """Generate dependency analysis section."""
        if not self.dependencies:
            return "No dependency information found."

        sections = []

        if "dependencies" in self.dependencies:
            sections.append("### 📦 Production Dependencies")
            for dep in self.dependencies["dependencies"]:
                sections.append(f"- `{dep}`")
            sections.append("")

        if "dev_dependencies" in self.dependencies:
            sections.append("### 🔧 Development Dependencies")
            for category, deps in self.dependencies["dev_dependencies"].items():
                sections.append(f"**{category}:**")
                if isinstance(deps, list):
                    for dep in deps:
                        sections.append(f"- `{dep}`")
            sections.append("")

        if "tool_config" in self.dependencies:
            tools = list(self.dependencies["tool_config"].keys())
            if tools:
                sections.append("### ⚙️ Configured Tools")
                sections.append(", ".join(f"`{tool}`" for tool in tools))
                sections.append("")

        return "\n".join(sections)

    def _generate_architecture_flow(self) -> str:
        """Generate architecture flow from call graph analysis."""
        sections = []
        sections.append("### 🔄 Import Dependencies Flow")

        # Find main entry points
        main_files = [f for f in self.call_graph.keys() if "main.py" in f]

        if main_files:
            sections.append("**Main Entry Points:**")
            for main_file in main_files:
                sections.append(f"- `{main_file}`")
                if main_file in self.call_graph:
                    for import_name in self.call_graph[main_file][:5]:  # Limit to first 5
                        sections.append(f"  └── {import_name}")
            sections.append("")

        # Orchestrator flow if detected
        orchestrator_files = [f for f in self.call_graph.keys() if "orchestrator" in f]
        if orchestrator_files:
            sections.append("**Orchestrators Chain:**")
            for orch_file in orchestrator_files:
                sections.append(f"- `{orch_file}`")

        return "\n".join(sections)

    def _generate_docstring_summary(self) -> str:
        """Generate summary of key docstrings."""
        sections = []
        sections.append("### 📚 Key Components Documentation")

        for file_path, docs in self.docstrings.items():
            if "module" in docs:
                sections.append(f"**{file_path}:**")
                sections.append(f"{docs['module']}")
                sections.append("")

        return "\n".join(sections)

    def generate_context_md(self, output_path: str | None = None) -> Path:
        """Generate comprehensive session context markdown file."""
        # Perform analysis first
        self.analyze_project()

        if output_path is None:
            output_path = self.DEFAULT_FILENAME

        context_content = self._build_comprehensive_template()

        file_path = Path(output_path)
        file_path.write_text(context_content, encoding="utf-8")

        self.logger.info(f"Session context generated: {file_path}")
        return file_path.resolve()

    def _build_comprehensive_template(self) -> str:
        """Build comprehensive context template with all analysis results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f"""# {self.project_name} Project Session Context

## 📊 Project Overview
- **Architecture**: Automated analysis of modular ML pipeline
- **Last Updated**: {timestamp}
- **Files Analyzed**: {len(self.python_files)} Python files, {len(self.config_files)} config files
- **Root Path**: `{self.root_path.resolve()}`

## 🎯 Automated Analysis Results

{self._generate_file_listing()}

{self._generate_dependency_analysis()}

{self._generate_architecture_flow()}

{self._generate_docstring_summary()}

## 🔧 Coding Standards (Auto-Applied)
- **Style**: PEP8 + Ruff strict compliance
- **Docstrings**: Mandatory (summary + detailed description)
- **Imports**: Ordered (stdlib → third-party → local)
- **Constants**: UPPERCASE at class top
- **Types**: PEP 604 unions (X | None), builtin generics (dict/list/tuple)
- **Format**: Production-ready, copy-pastable code blocks

## 📁 Project Structure Analysis
{self.project_name}/
├── Python files: {len(self.python_files)}
├── Config files: {len(self.config_files)}
├── Total modules with docs: {len([f for f in self.docstrings.values() if "module" in f])}
└── Dependencies analyzed: {len(self.dependencies.get("dependencies", []))}

text

## 🚀 Current State (Auto-Detected)
- ✅ Files discovered and analyzed automatically
- ✅ Dependencies extracted from configuration
- ✅ Docstrings catalogued and summarized  
- ✅ Import relationships mapped
- ✅ Architecture flow documented

## 📋 Session Usage
1. **Start new chat** with: "Context: MLP project from session-context.md"
2. **Attach this file** to provide immediate context
3. **Mention specific files** from the analysis above
4. **Standards auto-applied** - no need to re-specify

## 🎯 Analysis Insights
- **Most documented module**: {self._find_most_documented_module()}
- **Main entry points**: {len([f for f in self.call_graph.keys() if "main" in f])} detected
- **Orchestrator pattern**: {"✅ Detected" if any("orchestrator" in str(f) for f in self.python_files) else "❌ Not found"}

---
*Generated by AdvancedSessionContextGenerator v{self.TEMPLATE_VERSION}*
*Automated analysis of {len(self.python_files)} files completed at {timestamp}*
"""

    def _find_most_documented_module(self) -> str:
        """Find the module with the most comprehensive documentation."""
        max_docs = 0
        best_module = "None"

        for file_path, docs in self.docstrings.items():
            doc_count = len(docs)
            if doc_count > max_docs:
                max_docs = doc_count
                best_module = file_path

        return best_module if max_docs > 1 else "None found"


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    generator = AdvancedSessionContextGenerator("MLP", ".")
    context_file = generator.generate_context_md()
    print(f"✅ Contexte intelligent généré: {context_file}")
    print(f"📊 Analysé: {len(generator.python_files)} fichiers Python")
    print(f"📦 Dépendances: {len(generator.dependencies.get('dependencies', []))}")
    print("📎 Attachez ce fichier à vos nouveaux fils ChatGPT")
