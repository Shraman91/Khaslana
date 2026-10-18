"""Codebase and fix interaction style learner for personalized AI assistance."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from khaslana.memory.models import FixRecord, StyleProfile, StyleRule


class StyleLearner:
    """Analyzes AST structures and user-accepted patches to infer coding style patterns."""

    def analyze_codebase(self, root_dir: Path | str, max_files: int = 50) -> StyleProfile:
        """Inspect source files in codebase and generate a comprehensive StyleProfile."""
        root = Path(root_dir).resolve()
        py_files: list[Path] = []

        ignore_dirs = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            ".khaslana",
        }

        for path in root.rglob("*.py"):
            if any(part in ignore_dirs for part in path.parts):
                continue
            py_files.append(path)
            if len(py_files) >= max_files:
                break

        if not py_files:
            return StyleProfile()

        total_funcs = 0
        typed_funcs = 0
        docstring_funcs = 0
        snake_case_funcs = 0
        camel_case_funcs = 0
        pascal_case_classes = 0
        total_classes = 0

        google_docstrings = 0
        sphinx_docstrings = 0
        framework_hits: set[str] = set()

        for file_path in py_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content)
            except Exception:
                continue

            # Check framework signatures in imports
            if "from fastapi" in content or "import fastapi" in content:
                framework_hits.add("FastAPI router/endpoint patterns")
            if "from typer" in content or "import typer" in content:
                framework_hits.add("Typer CLI commands with Annotated options")
            if "pydantic" in content:
                framework_hits.add("Pydantic v2 BaseModel schemas")
            if "pytest" in content:
                framework_hits.add("Pytest test fixtures and assertions")

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    name = node.name
                    if not name.startswith("__"):
                        if "_" in name or name.islower():
                            snake_case_funcs += 1
                        elif any(c.isupper() for c in name):
                            camel_case_funcs += 1

                    # Typing analysis
                    has_return = node.returns is not None
                    annotated_args = sum(1 for arg in node.args.args if arg.annotation is not None)
                    total_args = len(node.args.args)
                    if (
                        total_args > 0
                        and annotated_args == total_args
                    ) or has_return:
                        typed_funcs += 1

                    # Docstring analysis
                    doc = ast.get_docstring(node)
                    if doc:
                        docstring_funcs += 1
                        if "Args:" in doc or "Returns:" in doc:
                            google_docstrings += 1
                        elif ":param" in doc or ":return:" in doc:
                            sphinx_docstrings += 1

                elif isinstance(node, ast.ClassDef):
                    total_classes += 1
                    name = node.name
                    if name[0].isupper() and "_" not in name:
                        pascal_case_classes += 1

        # Synthesize naming convention
        naming = "snake_case functions and variables, PascalCase classes"
        if camel_case_funcs > snake_case_funcs:
            naming = "camelCase functions and variables, PascalCase classes"

        # Synthesize typing discipline
        typing_ratio = typed_funcs / max(1, total_funcs)
        if typing_ratio >= 0.7:
            typing = "Strict type hints required on all function arguments and returns"
        elif typing_ratio >= 0.3:
            typing = "Moderate type hints on public interfaces"
        else:
            typing = "Dynamic / minimal type annotations"

        # Synthesize docstring style
        doc_ratio = docstring_funcs / max(1, total_funcs)
        if google_docstrings >= sphinx_docstrings and google_docstrings > 0:
            doc_style = "Google-style docstrings (Args:, Returns:, Raises:)"
        elif sphinx_docstrings > 0:
            doc_style = "Sphinx/reST-style docstrings (:param, :return:)"
        elif doc_ratio > 0.5:
            doc_style = "Concise triple-quoted docstrings on public members"
        else:
            doc_style = "Minimal docstrings, self-documenting code"

        rules: list[StyleRule] = []
        rules.append(
            StyleRule(
                name="Naming",
                description=naming,
                pattern_type="naming",
                confidence=round(
                    (snake_case_funcs or camel_case_funcs) / max(1, total_funcs), 2
                ),
            )
        )
        rules.append(
            StyleRule(
                name="Type Annotations",
                description=typing,
                pattern_type="typing",
                confidence=round(typing_ratio, 2),
            )
        )
        rules.append(
            StyleRule(
                name="Documentation",
                description=doc_style,
                pattern_type="docstring",
                confidence=round(doc_ratio, 2),
            )
        )

        return StyleProfile(
            naming_convention=naming,
            typing_discipline=typing,
            docstring_style=doc_style,
            framework_patterns=sorted(framework_hits),
            rules=rules,
        )

    def update_from_accepted_fix(
        self, fix: FixRecord, current_profile: StyleProfile
    ) -> StyleProfile:
        """Refine style profile based on an accepted code patch."""
        profile_dict: dict[str, Any] = current_profile.model_dump()

        try:
            tree = ast.parse(fix.fixed_code)
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.returns is not None
                ):
                    # Reinforce strict typing rule
                    for r in profile_dict.get("rules", []):
                        if r.get("pattern_type") == "typing":
                            r["confidence"] = min(1.0, float(r.get("confidence", 0.5)) + 0.05)
        except Exception:
            pass

        return StyleProfile.model_validate(profile_dict)
