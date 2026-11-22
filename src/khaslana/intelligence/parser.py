"""AST and multi-language parser extracting symbols, imports, and call sites.

Note: Static analysis resolves call sites syntactically. Dynamic dispatch,
polymorphic calls, and runtime monkeypatching are approximate: callee targets
are matched by identifier name rather than full runtime type resolution.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Literal

from khaslana.intelligence.models import CallEdge, ImportEdge, ParsedFile, Symbol, SymbolType

logger = logging.getLogger(__name__)

# Flag to ensure the warning about missing tree-sitter is emitted at most once per process
_TREE_SITTER_WARNED: bool = False

KEYWORDS: set[str] = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "function",
    "pub",
    "fn",
    "let",
    "var",
    "const",
    "new",
    "type",
    "struct",
    "class",
    "interface",
    "impl",
    "package",
    "import",
    "def",
    "lambda",
    "async",
    "await",
    "synchronized",
    "throw",
    "throws",
    "case",
    "default",
    "else",
    "trait",
    "enum",
    "record",
    "try",
    "finally",
    "public",
    "private",
    "protected",
    "static",
    "final",
    "abstract",
    "override",
    "extends",
    "implements",
}

EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


def _warn_tree_sitter_once() -> None:
    """Emit a single warning when falling back from tree-sitter to regex."""
    global _TREE_SITTER_WARNED
    if not _TREE_SITTER_WARNED:
        logger.warning(
            "tree-sitter is not available or grammar missing; falling back to regex parser. "
            "Install parser dependencies with: pip install khaslana[parsers]"
        )
        _TREE_SITTER_WARNED = True


def _find_matching_delimiter(
    text: str, open_idx: int, open_char: str = "{", close_char: str = "}"
) -> int:
    """Find index of matching closing delimiter, respecting strings and comments."""
    depth = 0
    i = open_idx
    n = len(text)
    in_string: str | None = None
    in_line_comment = False
    in_block_comment = False

    while i < n:
        c = text[i]
        if not in_string:
            if not in_line_comment and not in_block_comment:
                if c == "/" and i + 1 < n:
                    if text[i + 1] == "/":
                        in_line_comment = True
                        i += 2
                        continue
                    elif text[i + 1] == "*":
                        in_block_comment = True
                        i += 2
                        continue
            elif in_line_comment:
                if c == "\n":
                    in_line_comment = False
                i += 1
                continue
            elif in_block_comment:
                if c == "*" and i + 1 < n and text[i + 1] == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

        if not in_line_comment and not in_block_comment:
            if in_string:
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == in_string:
                    in_string = None
                i += 1
                continue
            else:
                if c in ('"', "'", "`"):
                    in_string = c
                    i += 1
                    continue

        if not in_string and not in_line_comment and not in_block_comment:
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _line_no(text: str, offset: int) -> int:
    """Compute 1-based line number for character offset in text."""
    return text.count("\n", 0, offset) + 1


def _find_container_body_brace(content: str, after_name_idx: int) -> int:
    """Find the opening brace of a class/interface/struct/impl, skipping generics."""
    idx = after_name_idx
    n = len(content)
    while idx < n and content[idx].isspace():
        idx += 1
    if idx < n and content[idx] == "<":
        close_angle = _find_matching_delimiter(content, idx, "<", ">")
        if close_angle != -1:
            idx = close_angle + 1
    elif idx < n and content[idx] == "[":
        close_bracket = _find_matching_delimiter(content, idx, "[", "]")
        if close_bracket != -1:
            idx = close_bracket + 1

    return content.find("{", idx)


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
            return node.attr
        return None


def parse_python_source(file_path: str, content: str) -> ParsedFile:
    """Parse Python source code using stdlib AST.

    Note: Static analysis resolves call sites syntactically. Dynamic dispatch,
    polymorphic calls, and runtime monkeypatching are approximate: callee targets
    are matched by identifier name rather than full runtime type resolution.
    """
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
        return parse_generic_source(file_path, content, language="python")


def parse_tree_sitter(file_path: str, content: str, language: str) -> ParsedFile | None:
    """Parse source code using tree-sitter if installed and language grammar is available.

    Returns None if tree-sitter or the specific grammar package is unavailable.
    """
    try:
        import tree_sitter  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        lang_obj: Any = None
        if language in ("typescript", "ts"):
            import tree_sitter_typescript as ts  # type: ignore[import-not-found]

            lang_obj = tree_sitter.Language(ts.language_typescript())
        elif language in ("javascript", "js"):
            import tree_sitter_javascript as js  # type: ignore[import-not-found]

            lang_obj = tree_sitter.Language(js.language())
        elif language == "go":
            import tree_sitter_go as tsg  # type: ignore[import-not-found]

            lang_obj = tree_sitter.Language(tsg.language())
        elif language == "rust":
            import tree_sitter_rust as tsr  # type: ignore[import-not-found]

            lang_obj = tree_sitter.Language(tsr.language())
        elif language == "java":
            import tree_sitter_java as tsj  # type: ignore[import-not-found]

            lang_obj = tree_sitter.Language(tsj.language())

        if lang_obj is None:
            return None

        parser = tree_sitter.Parser(lang_obj)
        tree = parser.parse(bytes(content, "utf8"))
        root = tree.root_node

        symbols: list[Symbol] = []
        imports: list[ImportEdge] = []
        calls: list[CallEdge] = []

        def _traverse(node: Any, current_scope: str | None = None) -> None:
            nonlocal symbols, imports, calls
            ntype = node.type
            name: str | None = None
            sym_type: SymbolType | None = None

            if ntype in (
                "class_declaration",
                "class_definition",
                "interface_declaration",
                "struct_item",
                "type_declaration",
            ):
                sym_type = "interface" if "interface" in ntype else "class"
                name_node = node.child_by_field_name("name")
                if name_node and name_node.text:
                    name = name_node.text.decode("utf8")
                    start_line = node.start_point.row + 1
                    end_line = node.end_point.row + 1
                    symbols.append(
                        Symbol(
                            name=name,
                            symbol_type=sym_type,
                            file=file_path,
                            line_range=(start_line, end_line),
                            parent_symbol=current_scope,
                        )
                    )
                    for child in node.children:
                        _traverse(child, current_scope=name)
                    return

            elif ntype in (
                "function_declaration",
                "function_definition",
                "method_declaration",
                "method_definition",
                "function_item",
            ):
                sym_type = "method" if current_scope else "function"
                name_node = node.child_by_field_name("name")
                if name_node and name_node.text:
                    name = name_node.text.decode("utf8")
                    start_line = node.start_point.row + 1
                    end_line = node.end_point.row + 1
                    symbols.append(
                        Symbol(
                            name=name,
                            symbol_type=sym_type,
                            file=file_path,
                            line_range=(start_line, end_line),
                            parent_symbol=current_scope,
                        )
                    )
                    for child in node.children:
                        _traverse(child, current_scope=name)
                    return

            elif ntype in ("call_expression",):
                fn_node = node.child_by_field_name("function")
                if fn_node and fn_node.text:
                    callee = fn_node.text.decode("utf8").split(".")[-1]
                    calls.append(
                        CallEdge(
                            caller_file=file_path,
                            caller_symbol=current_scope or "<file_scope>",
                            callee_name=callee,
                            line_number=node.start_point.row + 1,
                        )
                    )

            for child in node.children:
                _traverse(child, current_scope)

        _traverse(root)
        return ParsedFile(
            file=file_path,
            language=language,
            symbols=symbols,
            imports=imports,
            calls=calls,
        )
    except Exception:
        return None


def parse_generic_source(file_path: str, content: str, language: str = "plaintext") -> ParsedFile:
    """Robust multi-language regex parser handling generics, multi-line signatures, and scopes.

    Note: Static analysis resolves call sites syntactically. Dynamic dispatch,
    polymorphic calls, and runtime monkeypatching are approximate: callee targets
    are matched by identifier name rather than full runtime type resolution.
    """
    symbols: list[Symbol] = []
    imports: list[ImportEdge] = []
    calls: list[CallEdge] = []
    scopes: list[dict[str, Any]] = []

    # 1. Imports
    # TS / JS imports
    for m in re.finditer(
        r"import\s+(?:(?:type\s+)?[\w\s{},*]+from\s+)?['\"]([^'\"]+)['\"]", content
    ):
        imports.append(
            ImportEdge(
                file=file_path,
                imported_module=m.group(1),
                line_number=_line_no(content, m.start()),
            )
        )
    for m in re.finditer(
        r"(?:const|let|var)\s+[\w\s{},:]+=\s*require\(['\"]([^'\"]+)['\"]\)", content
    ):
        imports.append(
            ImportEdge(
                file=file_path,
                imported_module=m.group(1),
                line_number=_line_no(content, m.start()),
            )
        )

    # Go imports: single or block
    for m in re.finditer(r'import\s+["\']([^"\']+)["\']', content):
        imports.append(
            ImportEdge(
                file=file_path,
                imported_module=m.group(1),
                line_number=_line_no(content, m.start()),
            )
        )
    for m in re.finditer(r"import\s*\(([^)]+)\)", content):
        block = m.group(1)
        block_start = m.start(1)
        for sub_m in re.finditer(r'["\']([^"\']+)["\']', block):
            line = _line_no(content, block_start + sub_m.start())
            imports.append(
                ImportEdge(file=file_path, imported_module=sub_m.group(1), line_number=line)
            )

    # Rust imports (use ...)
    for m in re.finditer(r"use\s+([A-Za-z0-9_:]+(?:::\{[^}]+\})?);", content):
        imports.append(
            ImportEdge(
                file=file_path,
                imported_module=m.group(1),
                line_number=_line_no(content, m.start()),
            )
        )

    # Java imports
    for m in re.finditer(r"import\s+(?:static\s+)?([A-Za-z0-9_.*]+);", content):
        imports.append(
            ImportEdge(
                file=file_path,
                imported_module=m.group(1),
                line_number=_line_no(content, m.start()),
            )
        )

    # 2. Container Scopes (Class, Interface, Struct, Trait, Impl)
    # Rust impl blocks
    for m in re.finditer(
        r"\bimpl(?:<[^>]*>)?\s+(?:[A-Za-z0-9_:]+(?:<[^>]*>)?\s+for\s+)?([A-Za-z0-9_]+)",
        content,
    ):
        name = m.group(1)
        brace_pos = _find_container_body_brace(content, m.end())
        if brace_pos != -1:
            close_idx = _find_matching_delimiter(content, brace_pos)
            if close_idx != -1:
                scopes.append(
                    {
                        "name": name,
                        "symbol_type": "class",
                        "start_idx": m.start(),
                        "end_idx": close_idx,
                    }
                )

    # Rust traits and structs
    for m in re.finditer(
        r"(?:pub(?:\([^)]+\))?\s+)?\b(struct|trait|enum)\s+([A-Za-z0-9_]+)", content
    ):
        sym_type: Literal["class", "interface"] = "interface" if m.group(1) == "trait" else "class"
        name = m.group(2)
        start_line = _line_no(content, m.start())
        brace_pos = _find_container_body_brace(content, m.end())
        semi_pos = content.find(";", m.end())
        if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
            close_idx = _find_matching_delimiter(content, brace_pos)
            end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
            scopes.append(
                {
                    "name": name,
                    "symbol_type": sym_type,
                    "start_idx": m.start(),
                    "end_idx": close_idx if close_idx != -1 else m.end(),
                }
            )
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type=sym_type,
                    file=file_path,
                    line_range=(start_line, end_line),
                )
            )
        else:
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type=sym_type,
                    file=file_path,
                    line_range=(start_line, start_line),
                )
            )

    # Rust macros
    for m in re.finditer(r"\bmacro_rules!\s+([A-Za-z0-9_]+)", content):
        start_line = _line_no(content, m.start())
        brace_pos = content.find("{", m.end())
        close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
        end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        symbols.append(
            Symbol(
                name=m.group(1),
                symbol_type="function",
                file=file_path,
                line_range=(start_line, end_line),
            )
        )

    # Go type ... struct/interface
    for m in re.finditer(r"\btype\s+([A-Za-z0-9_]+)(?:\[[^\]]+\])?\s+(struct|interface)", content):
        sym_type = "interface" if m.group(2) == "interface" else "class"
        name = m.group(1)
        start_line = _line_no(content, m.start())
        brace_pos = content.find("{", m.end())
        close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
        end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        if close_idx != -1:
            scopes.append(
                {
                    "name": name,
                    "symbol_type": sym_type,
                    "start_idx": m.start(),
                    "end_idx": close_idx,
                }
            )
        symbols.append(
            Symbol(
                name=name,
                symbol_type=sym_type,
                file=file_path,
                line_range=(start_line, end_line),
            )
        )

    # TS / JS / Java classes, interfaces, records
    for m in re.finditer(
        r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:public|private|protected|static|final|\s)*\b(class|interface|record)\s+([A-Za-z0-9_$]+)",
        content,
    ):
        name = m.group(2)
        if name in KEYWORDS:
            continue
        sym_type = "interface" if m.group(1) == "interface" else "class"
        start_line = _line_no(content, m.start())
        brace_pos = _find_container_body_brace(content, m.end())
        close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
        end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        if close_idx != -1:
            scopes.append(
                {
                    "name": name,
                    "symbol_type": sym_type,
                    "start_idx": m.start(),
                    "end_idx": close_idx,
                }
            )
        symbols.append(
            Symbol(
                name=name,
                symbol_type=sym_type,
                file=file_path,
                line_range=(start_line, end_line),
            )
        )

    def get_parent_scope(pos: int) -> dict[str, Any] | None:
        matching = [s for s in scopes if s["start_idx"] <= pos <= s["end_idx"]]
        if not matching:
            return None
        return min(matching, key=lambda s: s["end_idx"] - s["start_idx"])

    # 3. Functions & Methods
    # Go receiver methods
    for m in re.finditer(
        r"\bfunc\s*\(\s*(?:[A-Za-z0-9_]+\s+)?(?:\*\s*)?([A-Za-z0-9_]+)\s*\)\s*([A-Za-z0-9_]+)\s*(?:\[[^\]]+\])?\s*\(",
        content,
    ):
        receiver = m.group(1)
        fn_name = m.group(2)
        start_line = _line_no(content, m.start())
        paren_close = _find_matching_delimiter(content, m.end() - 1, "(", ")")
        brace_pos = (
            content.find("{", paren_close) if paren_close != -1 else content.find("{", m.end())
        )
        close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
        end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        symbols.append(
            Symbol(
                name=fn_name,
                symbol_type="method",
                file=file_path,
                line_range=(start_line, end_line),
                parent_symbol=receiver,
            )
        )

    # Go standalone functions
    for m in re.finditer(r"\bfunc\s+([A-Za-z0-9_]+)\s*(?:\[[^\]]+\])?\s*\(", content):
        fn_name = m.group(1)
        start_line = _line_no(content, m.start())
        paren_close = _find_matching_delimiter(content, m.end() - 1, "(", ")")
        brace_pos = (
            content.find("{", paren_close) if paren_close != -1 else content.find("{", m.end())
        )
        close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
        end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        symbols.append(
            Symbol(
                name=fn_name,
                symbol_type="function",
                file=file_path,
                line_range=(start_line, end_line),
                parent_symbol=None,
            )
        )

    # Rust fn
    for m in re.finditer(
        r"(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern(?:\s+\"[A-Za-z]+\")?\s+)?fn\s+([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\(",
        content,
    ):
        fn_name = m.group(1)
        start_line = _line_no(content, m.start())
        paren_close = _find_matching_delimiter(content, m.end() - 1, "(", ")")
        brace_pos = content.find("{", paren_close) if paren_close != -1 else -1
        semi_pos = content.find(";", paren_close) if paren_close != -1 else -1
        if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
            close_idx = _find_matching_delimiter(content, brace_pos)
            end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        else:
            end_line = _line_no(content, semi_pos) if semi_pos != -1 else start_line

        parent = get_parent_scope(m.start())
        parent_name = parent["name"] if parent else None
        fn_sym_type: Literal["function", "method"] = "method" if parent_name else "function"
        symbols.append(
            Symbol(
                name=fn_name,
                symbol_type=fn_sym_type,
                file=file_path,
                line_range=(start_line, end_line),
                parent_symbol=parent_name,
            )
        )

    # TS arrow functions: (export )?(const|let|var) foo = async? <...>? (...): Ret => {
    for m in re.finditer(
        r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?", content
    ):
        fn_name = m.group(1)
        idx = m.end()
        # Skip generics if present
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx < len(content) and content[idx] == "<":
            close_angle = _find_matching_delimiter(content, idx, "<", ">")
            if close_angle != -1:
                idx = close_angle + 1
        paren_open = content.find("(", idx)
        if paren_open != -1 and paren_open - idx < 50:
            paren_close = _find_matching_delimiter(content, paren_open, "(", ")")
            if paren_close != -1:
                arrow_pos = content.find("=>", paren_close)
                if arrow_pos != -1 and arrow_pos - paren_close < 300:
                    start_line = _line_no(content, m.start())
                    brace_pos = content.find("{", arrow_pos)
                    close_idx = (
                        _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
                    )
                    end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
                    symbols.append(
                        Symbol(
                            name=fn_name,
                            symbol_type="function",
                            file=file_path,
                            line_range=(start_line, end_line),
                            parent_symbol=None,
                        )
                    )

    # TS / JS standard functions
    for m in re.finditer(
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*(?:<[^>]*>)?\s*\(",
        content,
    ):
        fn_name = m.group(1)
        start_line = _line_no(content, m.start())
        paren_close = _find_matching_delimiter(content, m.end() - 1, "(", ")")
        brace_pos = (
            content.find("{", paren_close) if paren_close != -1 else content.find("{", m.end())
        )
        close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
        end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
        symbols.append(
            Symbol(
                name=fn_name,
                symbol_type="function",
                file=file_path,
                line_range=(start_line, end_line),
                parent_symbol=None,
            )
        )

    # Methods with modifiers (public/private/async/static etc.)
    method_decl_re = re.compile(
        r"(?:(?:public|private|protected|readonly|static|async|override|synchronized|default|abstract)\s+)+"
        r"(?:<[^>]*>\s+)?"
        r"(?:[\w<>\[\],\s\?|&]+?\s+)?"
        r"([A-Za-z0-9_$]+)\s*(?:<[^>]*>)?\s*\("
    )
    for m in method_decl_re.finditer(content):
        name = m.group(1)
        start_idx = m.start(1)
        if name in KEYWORDS:
            continue
        parent = get_parent_scope(start_idx)
        if not parent:
            continue
        prefix = content[max(0, m.start() - 10) : m.start()].strip()
        if prefix.endswith("=") or prefix.endswith("."):
            continue
        start_line = _line_no(content, start_idx)
        paren_close = _find_matching_delimiter(content, m.end() - 1, "(", ")")
        end_line = start_line
        if paren_close != -1:
            brace_pos = content.find("{", paren_close)
            semi_pos = content.find(";", paren_close)
            if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
                close_idx = _find_matching_delimiter(content, brace_pos)
                if close_idx != -1:
                    end_line = _line_no(content, close_idx)
            elif semi_pos != -1:
                end_line = _line_no(content, semi_pos)

        symbols.append(
            Symbol(
                name=name,
                symbol_type="method",
                file=file_path,
                line_range=(start_line, end_line),
                parent_symbol=parent["name"],
            )
        )

    # Class methods without modifiers (e.g., getUser(id: string) { or constructor(...) {)
    for m in re.finditer(
        r"(?:^|[\n;{}])\s*([A-Za-z0-9_$]+)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^;{]+)?\s*\{",
        content,
    ):
        name = m.group(1)
        if name in KEYWORDS:
            continue
        parent = get_parent_scope(m.start(1))
        if parent:
            start_line = _line_no(content, m.start(1))
            brace_pos = content.find("{", m.end() - 1)
            close_idx = _find_matching_delimiter(content, brace_pos) if brace_pos != -1 else -1
            end_line = _line_no(content, close_idx) if close_idx != -1 else start_line
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="method",
                    file=file_path,
                    line_range=(start_line, end_line),
                    parent_symbol=parent["name"],
                )
            )

    # Interface methods ending in ';'
    for m in re.finditer(
        r"\b([A-Za-z0-9_$]+)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^;{]+)?\s*;", content
    ):
        name = m.group(1)
        if name in KEYWORDS:
            continue
        parent = get_parent_scope(m.start())
        if parent and parent["symbol_type"] == "interface":
            line = _line_no(content, m.start())
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type="method",
                    file=file_path,
                    line_range=(line, line),
                    parent_symbol=parent["name"],
                )
            )

    # Deduplicate symbols
    unique_symbols: list[Symbol] = []
    seen: set[tuple[str, str, str | None, int]] = set()
    for s in symbols:
        key = (s.name, s.symbol_type, s.parent_symbol, s.line_range[0])
        if key not in seen:
            seen.add(key)
            unique_symbols.append(s)

    # 4. Calls
    def get_enclosing_function(line: int) -> str:
        candidates = [
            s
            for s in unique_symbols
            if s.symbol_type in ("function", "method")
            and s.line_range[0] <= line <= s.line_range[1]
        ]
        if not candidates:
            return "<file_scope>"
        innermost = min(candidates, key=lambda s: s.line_range[1] - s.line_range[0])
        return (
            f"{innermost.parent_symbol}.{innermost.name}"
            if innermost.parent_symbol
            else innermost.name
        )

    for m in re.finditer(r"\b([A-Za-z0-9_$]+)\s*\(", content):
        callee = m.group(1)
        if callee in KEYWORDS or callee == "func":
            continue
        line = _line_no(content, m.start())
        caller = get_enclosing_function(line)
        calls.append(
            CallEdge(
                caller_file=file_path,
                caller_symbol=caller,
                callee_name=callee,
                line_number=line,
            )
        )

    for m in re.finditer(r"\b([A-Za-z0-9_]+)!\s*[\(\{\[]", content):
        callee = m.group(1)
        if callee in KEYWORDS or callee == "macro_rules":
            continue
        line = _line_no(content, m.start())
        caller = get_enclosing_function(line)
        calls.append(
            CallEdge(
                caller_file=file_path,
                caller_symbol=caller,
                callee_name=callee,
                line_number=line,
            )
        )

    return ParsedFile(
        file=file_path,
        language=language,
        symbols=unique_symbols,
        imports=imports,
        calls=calls,
    )


def parse_source(file_path: str, content: str, language: str = "python") -> ParsedFile:
    """Unified entry point to parse a source code file into a ParsedFile.

    Note: Static analysis resolves call sites syntactically. Dynamic dispatch,
    polymorphic calls, and runtime monkeypatching are approximate: callee targets
    are matched by identifier name rather than full runtime type resolution.
    """
    ext = Path(file_path).suffix.lower()
    target_lang = EXTENSION_TO_LANG.get(ext, language)

    if target_lang == "python" or file_path.endswith(".py"):
        return parse_python_source(file_path, content)

    # Multi-language parsing: try tree-sitter if installed, else fallback to regex
    ts_result = parse_tree_sitter(file_path, content, target_lang)
    if ts_result is not None:
        return ts_result

    _warn_tree_sitter_once()
    return parse_generic_source(file_path, content, target_lang)
