"""Unit tests for Code Intelligence Layer (Phase 3)."""

from pathlib import Path

from typer.testing import CliRunner

from khaslana.cli.main import app
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.models import CallEdge, ImportEdge, ParsedFile, Symbol
from khaslana.intelligence.parser import parse_source
from khaslana.intelligence.symbols import SymbolTable

runner = CliRunner()
SAMPLE_REPO_PATH = Path(__file__).parent / "fixtures" / "sample_repo"


def test_python_ast_parsing() -> None:
    code = '''"""Sample module docstring."""
import os
from math import sqrt as root

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def compute(self, x: int) -> int:
        val = self.add(x, 10)
        return root(val)

def helper_function():
    calc = Calculator()
    return calc.compute(5)
'''
    parsed = parse_source("calc.py", code, language="python")
    assert parsed.file == "calc.py"
    assert parsed.language == "python"

    # Symbols
    sym_names = {s.name for s in parsed.symbols}
    assert "Calculator" in sym_names
    assert "add" in sym_names
    assert "compute" in sym_names
    assert "helper_function" in sym_names

    # Check method parent symbol
    add_sym = next(s for s in parsed.symbols if s.name == "add")
    assert add_sym.parent_symbol == "Calculator"
    assert add_sym.symbol_type == "method"

    # Imports
    assert len(parsed.imports) == 2
    imp_math = next(i for i in parsed.imports if i.imported_module == "math")
    assert imp_math.imported_symbol == "sqrt"
    assert imp_math.alias == "root"

    # Calls
    call_names = {c.callee_name for c in parsed.calls}
    assert "add" in call_names
    assert "root" in call_names
    assert "Calculator" in call_names
    assert "compute" in call_names


def test_multilang_regex_parsing() -> None:
    ts_code = """
import { Request, Response } from 'express';

export class UserService {
    getUser(id: string) {
        return fetchUser(id);
    }
}

export function handleAuth(req: Request) {
    validateToken(req);
}
"""
    parsed = parse_source("service.ts", ts_code, language="typescript")
    assert parsed.language == "typescript"
    sym_names = {s.name for s in parsed.symbols}
    assert "UserService" in sym_names
    assert "getUser" in sym_names
    assert "handleAuth" in sym_names

    call_names = {c.callee_name for c in parsed.calls}
    assert "fetchUser" in call_names
    assert "validateToken" in call_names


def test_symbol_table_crud(tmp_path: Path) -> None:
    db_file = tmp_path / "symbols.db"
    sym_table = SymbolTable(db_file)

    parsed = ParsedFile(
        file="auth.py",
        language="python",
        symbols=[
            Symbol(
                name="AuthService",
                symbol_type="class",
                file="auth.py",
                line_range=(1, 20),
                docstring="Authentication service",
            ),
            Symbol(
                name="login",
                symbol_type="method",
                file="auth.py",
                line_range=(5, 10),
                parent_symbol="AuthService",
                signature="username, password",
            ),
        ],
        imports=[
            ImportEdge(
                file="auth.py",
                imported_module="jwt",
                imported_symbol="encode",
                line_number=2,
            )
        ],
        calls=[
            CallEdge(
                caller_file="auth.py",
                caller_symbol="login",
                callee_name="encode",
                line_number=8,
            )
        ],
    )

    sym_table.index_parsed_file(parsed)

    # Find definition
    defs = sym_table.find_definitions("AuthService")
    assert len(defs) == 1
    assert defs[0].name == "AuthService"
    assert defs[0].symbol_type == "class"

    method_defs = sym_table.find_definitions("login")
    assert len(method_defs) == 1
    assert method_defs[0].parent_symbol == "AuthService"

    # Find references
    refs = sym_table.find_references("encode")
    assert len(refs) == 1
    assert refs[0].caller_symbol == "login"
    assert refs[0].caller_file == "auth.py"

    # Overwrite test
    sym_table.index_parsed_file(parsed)
    assert len(sym_table.find_definitions("AuthService")) == 1


def test_callgraph_traversal_and_cycles(tmp_path: Path) -> None:
    db_file = tmp_path / "symbols.db"
    sym_table = SymbolTable(db_file)

    # Create a cyclic call graph: A -> B -> C -> A
    parsed_a = ParsedFile(
        file="a.py",
        language="python",
        symbols=[Symbol(name="fn_a", symbol_type="function", file="a.py", line_range=(1, 5))],
        imports=[],
        calls=[
            CallEdge(
                caller_file="a.py",
                caller_symbol="fn_a",
                callee_name="fn_b",
                line_number=3,
            )
        ],
    )
    parsed_b = ParsedFile(
        file="b.py",
        language="python",
        symbols=[Symbol(name="fn_b", symbol_type="function", file="b.py", line_range=(1, 5))],
        imports=[],
        calls=[
            CallEdge(
                caller_file="b.py",
                caller_symbol="fn_b",
                callee_name="fn_c",
                line_number=3,
            )
        ],
    )
    parsed_c = ParsedFile(
        file="c.py",
        language="python",
        symbols=[Symbol(name="fn_c", symbol_type="function", file="c.py", line_range=(1, 5))],
        imports=[],
        calls=[
            CallEdge(
                caller_file="c.py",
                caller_symbol="fn_c",
                callee_name="fn_a",
                line_number=3,
            )
        ],
    )

    sym_table.index_parsed_file(parsed_a)
    sym_table.index_parsed_file(parsed_b)
    sym_table.index_parsed_file(parsed_c)

    graph = CallGraph.from_symbol_table(sym_table)

    assert graph.get_callees("fn_a") == ["fn_b"]
    assert graph.get_callers("fn_b") == ["fn_a"]

    # Transitive with depth 2
    transitive_callees = graph.get_transitive_callees("fn_a", max_depth=2)
    assert "fn_b" in transitive_callees
    assert "fn_c" in transitive_callees

    # Cycle detection
    cycles = graph.find_cycles()
    assert len(cycles) > 0
    assert any("fn_a" in c for c in cycles)

    # Mermaid export
    mermaid = graph.export_mermaid("fn_a", depth=2)
    assert "graph TD" in mermaid
    assert "fn_a" in mermaid
    assert "fn_b" in mermaid

    # DOT export
    dot = graph.export_dot()
    assert "digraph CallGraph" in dot
    assert "fn_a -> fn_b" in dot


def test_where_used_and_graph_cli(tmp_path: Path) -> None:
    # First index the sample repo in a temp copy
    import shutil

    sample_copy = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO_PATH, sample_copy)

    # Run index command
    res_index = runner.invoke(app, ["index", str(sample_copy)])
    assert res_index.exit_code == 0

    # Test where-used command
    res_where = runner.invoke(app, ["where-used", "create_access_token"])
    assert res_where.exit_code == 0
    assert "Definitions" in res_where.output or "References" in res_where.output

    # Test graph command
    res_graph = runner.invoke(app, ["graph", "authenticate_user", "--format", "mermaid"])
    assert res_graph.exit_code == 0
    assert "graph TD" in res_graph.output

    # Test graph dot format
    res_dot = runner.invoke(app, ["graph", "authenticate_user", "--format", "dot"])
    assert res_dot.exit_code == 0
    assert "digraph" in res_dot.output
