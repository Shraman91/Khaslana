"""AST and multi-language parser extracting symbols, imports, and call sites."""

from __future__ import annotations

import ast
import re
from typing import Literal

from khaslana.intelligence.models import CallEdge, ImportEdge, ParsedFile, Symbol


class PythonASTVisitor(ast.NodeVisitor):
    """AST visitor extracting symbols, imports, and function calls from Python code."""

    def __init__(self, file_path: str, lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.symbols: list[Symbol] = []
        self.imports: list[ImportEdge] = []
        self.calls: list[CallEdge] = []
        self._current_scope: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportEdge(
                    file=self.file_path,
                    imported_module=alias.name,
                    alias=alias.asname,
                    line_number=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(
                ImportEdge(
                    file=self.file_path,
                    imported_module=mod,
                    imported_symbol=alias.name,
                    alias=alias.asname,
                    line_number=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        doc = ast.get_docstring(node)
        parent = ".".join(self._current_scope) if self._current_scope else None

        self.symbols.append(
            Symbol(
                name=node.name,
                symbol_type="class",
                file=self.file_path,
                line_range=(start, end),
                docstring=doc,
                parent_symbol=parent,
            )
        )

        self._current_scope.append(node.name)
        self.generic_visit(node)
        self._current_scope.pop()

    def _handle_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool = False
    ) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        doc = ast.get_docstring(node)
        parent = ".".join(self._current_scope) if self._current_scope else None
        sym_type: Literal["function", "method"] = (
            "method" if (self._current_scope or is_method) else "function"
        )

        # Extract arg names
        args = [a.arg for a in node.args.args]
        sig = f"({', '.join(args)})"

        self.symbols.append(
            Symbol(
                name=node.name,
                symbol_type=sym_type,
                file=self.file_path,
                line_range=(start, end),
                docstring=doc,
                parent_symbol=parent,
                signature=sig,
            )
        )

        full_sym_name = f"{parent}.{node.name}" if parent else node.name
        self._current_scope.append(node.name)

        # Inspect body for calls
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = self._resolve_callee_name(child.func)
                if callee_name:
                    self.calls.append(
                        CallEdge(
                            caller_file=self.file_path,
                            caller_symbol=full_sym_name,
                            callee_name=callee_name,
                            line_number=child.lineno,
                        )
                    )

        self._current_scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _resolve_callee_name(self, node: ast.AST) -> str | None:
        """Resolve called function name from Call.func."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # e.g., self.db.find_user -> find_user or db.find_user
            return node.attr
        return None


def parse_python_source(file_path: str, content: str) -> ParsedFile:
    """Parse Python source code using AST."""
    lines = content.splitlines(keepends=True)
    try:
        tree = ast.parse(content, filename=file_path)
        visitor = PythonASTVisitor(file_path, lines)
        visitor.visit(tree)
        return ParsedFile(
            file=file_path,
            language="python",
            symbols=visitor.symbols,
            imports=visitor.imports,
            calls=visitor.calls,
        )
    except Exception:
        # Fall back to regex parser if AST fails due to syntax error
        return parse_generic_source(file_path, content, language="python")


def parse_generic_source(
    file_path: str, content: str, language: str = "plaintext"
) -> ParsedFile:
    """Multi-language regex-based parser (JS/TS, Go, Rust, Java, C++)."""
    symbols: list[Symbol] = []
    imports: list[ImportEdge] = []
    calls: list[CallEdge] = []
    lines = content.splitlines()

    # Generic function / method patterns
    fn_patterns = [
        # JS/TS: function foo(), const foo = () =>, async foo()
        (r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(", "function"),
        (r"(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", "function"),
        # Go: func foo() or func (r *Receiver) Method()
        (r"func\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)\s*\(", "function"),
        # Rust: fn foo() or pub fn foo()
        (r"(?:pub\s+)?fn\s+([A-Za-z0-9_]+)\s*\(", "function"),
        # JS/TS class method: foo(arg: type) { or async foo() {
        (
            r"^(?:(?:public|private|protected|async|static)\s+)*([A-Za-z0-9_$]+)\s*\([^)]*\)\s*(?::\s*[^;{]+)?\s*\{",
            "method",
        ),
        # Java/C#: public void foo() or class Foo
        (
            r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{",
            "method",
        ),
    ]

    # Class patterns
    class_pattern = re.compile(
        r"(?:export\s+)?(?:class|struct|interface|type)\s+([A-Za-z0-9_$]+)", re.MULTILINE
    )

    for i, line in enumerate(lines, 1):
        line_str = line.strip()

        # Check classes
        cls_match = class_pattern.search(line_str)
        if cls_match:
            symbols.append(
                Symbol(
                    name=cls_match.group(1),
                    symbol_type="class",
                    file=file_path,
                    line_range=(i, i),
                )
            )

        # Check functions
        for pattern_str, stype in fn_patterns:
            fn_match = re.search(pattern_str, line_str)
            if fn_match:
                fn_name = fn_match.group(1)
                if fn_name in {"if", "for", "while", "switch", "catch", "return", "function"}:
                    continue
                symbols.append(
                    Symbol(
                        name=fn_name,
                        symbol_type=stype,  # type: ignore[arg-type]
                        file=file_path,
                        line_range=(i, i),
                    )
                )
                break

        # Check import lines
        # JS/TS: import { a } from 'b';
        # Go: import "b"
        # Rust: use a::b;
        if (
            line_str.startswith("import ")
            or line_str.startswith("use ")
            or line_str.startswith("require(")
        ):
            imports.append(
                ImportEdge(
                    file=file_path,
                    imported_module=line_str,
                    line_number=i,
                )
            )

        # Check function calls: name(...)
        call_matches = re.findall(r"\b([A-Za-z0-9_$]+)\s*\(", line_str)
        for callee in call_matches:
            if callee not in {"if", "for", "while", "switch", "catch", "return", "function"}:
                calls.append(
                    CallEdge(
                        caller_file=file_path,
                        caller_symbol="<file_scope>",
                        callee_name=callee,
                        line_number=i,
                    )
                )

    return ParsedFile(
        file=file_path,
        language=language,
        symbols=symbols,
        imports=imports,
        calls=calls,
    )


def parse_source(file_path: str, content: str, language: str = "python") -> ParsedFile:
    """Unified entry point to parse a source code file into a ParsedFile."""
    if language == "python" or file_path.endswith(".py"):
        return parse_python_source(file_path, content)
    return parse_generic_source(file_path, content, language)
