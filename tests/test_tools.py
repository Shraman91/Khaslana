"""Unit tests for Phase 4 Tool base classes and core built-in tools."""

from pathlib import Path

from pydantic import BaseModel, Field

from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.models import CallEdge, ParsedFile, Symbol
from khaslana.intelligence.symbols import SymbolTable
from khaslana.tools.base import Tool, ToolRegistry, ToolResult
from khaslana.tools.core import (
    FindSymbolTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    WhereUsedTool,
)


class DummyInput(BaseModel):
    message: str = Field(..., description="Message string")
    count: int = Field(default=1, description="Repeat count")


class DummyTool(Tool):
    name = "dummy_tool"
    description = "A dummy tool for registry testing."
    input_schema = DummyInput

    def execute(self, input: BaseModel) -> ToolResult:
        inp: DummyInput = input  # type: ignore[assignment]
        return ToolResult(ok=True, output=inp.message * inp.count)


def test_tool_registry_registration_and_execution() -> None:
    registry = ToolRegistry()
    dummy = DummyTool()
    registry.register(dummy)

    assert registry.get("dummy_tool") is dummy
    assert len(registry.list_tools()) == 1

    # Successful execution
    res = registry.execute("dummy_tool", {"message": "hi-", "count": 2})
    assert res.ok is True
    assert res.output == "hi-hi-"

    # Validation error
    bad_res = registry.execute("dummy_tool", {"message": 123, "count": "not-an-int"})
    assert bad_res.ok is False
    assert bad_res.error and "Invalid arguments" in bad_res.error

    # Unregistered tool
    missing_res = registry.execute("unknown_tool", {})
    assert missing_res.ok is False
    assert missing_res.error and "not registered" in missing_res.error

    # Descriptions formatting
    desc = registry.get_descriptions()
    assert "dummy_tool" in desc
    assert "message" in desc
    assert "count" in desc


def test_read_file_tool(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.py"
    test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")

    tool = ReadFileTool(root_dir=tmp_path)

    # Full read
    res = tool.execute(tool.input_schema.model_validate({"file_path": "sample.py"}))
    assert res.ok is True
    assert "1: line 1" in res.output
    assert "5: line 5" in res.output

    # Slice read
    slice_res = tool.execute(
        tool.input_schema.model_validate(
            {"file_path": "sample.py", "start_line": 2, "end_line": 4}
        )
    )
    assert slice_res.ok is True
    assert "2: line 2" in slice_res.output
    assert "4: line 4" in slice_res.output
    assert "1: line 1" not in slice_res.output

    # Missing file
    missing = tool.execute(tool.input_schema.model_validate({"file_path": "nonexistent.py"}))
    assert missing.ok is False
    assert missing.error and "File not found" in missing.error


def test_list_files_tool(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "b.py").write_text("y = 2")
    (tmp_path / ".gitignore").write_text("pkg/b.py\n")

    tool = ListFilesTool(root_dir=tmp_path)
    res = tool.execute(tool.input_schema.model_validate({"directory": "."}))
    assert res.ok is True
    assert "a.py" in res.output
    # b.py should be ignored according to .gitignore
    assert "pkg/b.py" not in res.output


def test_run_command_tool(tmp_path: Path) -> None:
    # Denied dangerous command
    tool = RunCommandTool(root_dir=tmp_path)
    denied = tool.execute(tool.input_schema.model_validate({"command": "rm -rf /"}))
    assert denied.ok is False
    assert denied.error and "blocked by security policy" in denied.error

    # Allowlisted command (e.g. echo)
    allowed = tool.execute(tool.input_schema.model_validate({"command": "echo test_cmd_pass"}))
    assert allowed.ok is True
    assert "test_cmd_pass" in allowed.output

    # Mutating command without confirmation handler
    mutating = tool.execute(tool.input_schema.model_validate({"command": "touch new_file.txt"}))
    assert mutating.ok is False
    assert mutating.error and "denied or unconfirmed" in mutating.error

    # Mutating command with confirmation handler accepting
    tool_with_handler = RunCommandTool(
        root_dir=tmp_path,
        confirmation_handler=lambda cmd: True,
    )
    confirmed = tool_with_handler.execute(
        tool.input_schema.model_validate({"command": "echo confirmed_run"})
    )
    assert confirmed.ok is True
    assert "confirmed_run" in confirmed.output


def test_symbol_tools(tmp_path: Path) -> None:
    db_file = tmp_path / "symbols.db"
    sym_table = SymbolTable(db_file)

    parsed = ParsedFile(
        file="service.py",
        language="python",
        symbols=[
            Symbol(
                name="PaymentProcessor",
                symbol_type="class",
                file="service.py",
                line_range=(1, 20),
                docstring="Processes payments.",
            ),
            Symbol(
                name="process",
                symbol_type="method",
                file="service.py",
                line_range=(5, 12),
                parent_symbol="PaymentProcessor",
                signature="amount: float",
            ),
        ],
        imports=[],
        calls=[
            CallEdge(
                caller_file="client.py",
                caller_symbol="main",
                callee_name="process",
                line_number=10,
            )
        ],
    )
    sym_table.index_parsed_file(parsed)
    call_graph = CallGraph(sym_table)

    find_tool = FindSymbolTool(sym_table)
    find_res = find_tool.execute(find_tool.input_schema.model_validate({"symbol_name": "process"}))
    assert find_res.ok is True
    assert "PaymentProcessor.process" in find_res.output
    assert "service.py:5-12" in find_res.output

    where_tool = WhereUsedTool(sym_table, call_graph)
    where_res = where_tool.execute(
        where_tool.input_schema.model_validate({"symbol_name": "process"})
    )
    assert where_res.ok is True
    assert "client.py" in where_res.output
    assert "main" in where_res.output
