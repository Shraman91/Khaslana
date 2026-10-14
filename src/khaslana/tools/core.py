"""Built-in core tools for Khaslana agent system."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.symbols import SymbolTable
from khaslana.retrieval.indexer import is_binary_file, load_gitignore_spec
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.tools.base import Tool, ToolResult

# Default safe inspection commands allowed without confirmation
DEFAULT_ALLOWLIST = {
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "pytest",
    "python -m pytest",
    "npm test",
    "cargo test",
    "ls",
    "dir",
    "cat",
    "echo",
    "pwd",
}

# Dangerous commands that are strictly prohibited
DEFAULT_DENYLIST = {
    "rm -rf",
    "format",
    "mkfs",
    "dd",
    "shutdown",
    "reboot",
    ":(){ :|:& };:",
}


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read (relative to project root)")
    start_line: int | None = Field(default=None, description="Starting line number (1-indexed)")
    end_line: int | None = Field(
        default=None, description="Ending line number (1-indexed, inclusive)"
    )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read contents of a file within the workspace. "
        "Optionally specify start_line and end_line to read a slice."
    )
    input_schema = ReadFileInput

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()

    def execute(self, input: BaseModel) -> ToolResult:
        inp: ReadFileInput = input  # type: ignore[assignment]
        target = (self.root_dir / inp.file_path).resolve()

        if not target.exists():
            return ToolResult(ok=False, output="", error=f"File not found: {inp.file_path}")
        if not target.is_file():
            return ToolResult(ok=False, output="", error=f"Path is not a file: {inp.file_path}")
        if is_binary_file(target):
            return ToolResult(
                ok=False, output="", error=f"Cannot read binary file: {inp.file_path}"
            )

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            start = max(1, inp.start_line) if inp.start_line is not None else 1
            end = min(total_lines, inp.end_line) if inp.end_line is not None else total_lines

            if start > total_lines:
                return ToolResult(
                    ok=True,
                    output=f"[Empty: start_line {start} exceeds total lines {total_lines}]",
                    metadata={"total_lines": total_lines},
                )

            selected_lines = lines[start - 1 : end]
            numbered = [f"{i}: {line}" for i, line in enumerate(selected_lines, start=start)]
            output = "\n".join(numbered)

            return ToolResult(
                ok=True,
                output=output,
                metadata={
                    "file_path": inp.file_path,
                    "total_lines": total_lines,
                    "start_line": start,
                    "end_line": end,
                },
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"Error reading {inp.file_path}: {e}")


class SearchCodeInput(BaseModel):
    query: str = Field(..., description="Semantic or keyword query to search across codebase")
    top_k: int = Field(default=5, description="Maximum number of code chunks to return")


class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "Search codebase using hybrid semantic and BM25 retrieval. "
        "Returns top matching code snippets with file and line range citations."
    )
    input_schema = SearchCodeInput

    def __init__(
        self,
        pipeline: RetrievalPipeline | None = None,
        root_dir: Path | str = ".",
    ) -> None:
        self.pipeline = pipeline
        self.root_dir = Path(root_dir).resolve()

    def execute(self, input: BaseModel) -> ToolResult:
        inp: SearchCodeInput = input  # type: ignore[assignment]
        if not self.pipeline:
            # Fallback or try to create memory/ephemeral pipeline if needed
            return ToolResult(
                ok=False,
                output="",
                error="Retrieval pipeline not initialized. Run index command first.",
            )

        try:
            results = self.pipeline.search(inp.query, top_k=inp.top_k)
            if not results:
                return ToolResult(
                    ok=True,
                    output=f"No matching code found for query: '{inp.query}'",
                    metadata={"count": 0},
                )

            formatted = []
            for i, res in enumerate(results, start=1):
                c = res.chunk
                loc = f"{c.file}:{c.line_range[0]}-{c.line_range[1]}"
                sym = f" [{c.symbol}]" if c.symbol else ""
                snippet = (
                    f"--- Result {i}: {loc}{sym} (Score: {res.score:.3f}) ---\n"
                    f"{c.text}\n"
                )
                formatted.append(snippet)

            return ToolResult(
                ok=True,
                output="\n".join(formatted),
                metadata={"count": len(results)},
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"Error searching code: {e}")


class ListFilesInput(BaseModel):
    directory: str = Field(default=".", description="Directory path relative to project root")
    max_depth: int = Field(default=3, description="Maximum depth for subdirectory traversal")


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files and directories in the workspace honoring .gitignore."
    input_schema = ListFilesInput

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()

    def execute(self, input: BaseModel) -> ToolResult:
        inp: ListFilesInput = input  # type: ignore[assignment]
        base_dir = (self.root_dir / inp.directory).resolve()

        if not base_dir.exists():
            return ToolResult(ok=False, output="", error=f"Directory not found: {inp.directory}")
        if not base_dir.is_dir():
            return ToolResult(
                ok=False, output="", error=f"Path is not a directory: {inp.directory}"
            )

        spec = load_gitignore_spec(self.root_dir)
        results: list[str] = []

        for dirpath, dirnames, filenames in os.walk(base_dir):
            rel_dir = Path(dirpath).relative_to(self.root_dir)
            depth = len(rel_dir.parts)

            # Filter out ignored directories
            dirnames[:] = [
                d
                for d in dirnames
                if not spec.match_file(str((rel_dir / d).as_posix()))
                and not spec.match_file(f"{d}/")
                and d not in {".git", "node_modules", "venv", ".venv", "__pycache__", ".khaslana"}
            ]

            if depth > inp.max_depth:
                continue

            for fname in filenames:
                rel_file = (rel_dir / fname).as_posix()
                if spec.match_file(rel_file) or spec.match_file(fname):
                    continue
                results.append(rel_file)

        results.sort()
        count = len(results)
        output = "\n".join(results) if results else "(No files found)"
        return ToolResult(ok=True, output=output, metadata={"file_count": count})


class RunCommandInput(BaseModel):
    command: str = Field(..., description="Shell command to execute in project directory")
    timeout_seconds: int = Field(default=30, description="Execution timeout in seconds")


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the project environment. "
        "Read-only commands run automatically; mutating commands require confirmation."
    )
    input_schema = RunCommandInput

    def __init__(
        self,
        root_dir: Path | str = ".",
        allowlist: set[str] | None = None,
        denylist: set[str] | None = None,
        confirmation_handler: Callable[[str], bool] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.allowlist = allowlist or set(DEFAULT_ALLOWLIST)
        self.denylist = denylist or set(DEFAULT_DENYLIST)
        self.confirmation_handler = confirmation_handler

    def _is_allowed(self, cmd: str) -> bool:
        cmd_clean = cmd.strip()
        for allowed in self.allowlist:
            if cmd_clean == allowed or cmd_clean.startswith(allowed + " "):
                return True
        return False

    def _is_denied(self, cmd: str) -> bool:
        cmd_clean = cmd.strip()
        return any(denied in cmd_clean for denied in self.denylist)

    def execute(self, input: BaseModel) -> ToolResult:
        inp: RunCommandInput = input  # type: ignore[assignment]
        cmd = inp.command.strip()

        # Check denylist
        if self._is_denied(cmd):
            return ToolResult(
                ok=False,
                output="",
                error=f"Command '{cmd}' is blocked by security policy.",
                metadata={"denied": True},
            )

        # Check allowlist; if not allowlisted, request confirmation
        if not self._is_allowed(cmd):
            if self.confirmation_handler is not None:
                allowed_by_user = self.confirmation_handler(cmd)
            else:
                # Default non-interactive policy: mutating commands without handler are blocked
                allowed_by_user = False

            if not allowed_by_user:
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"Execution of mutating command '{cmd}' was denied or unconfirmed.",
                    metadata={"confirmed": False},
                )

        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.root_dir),
                capture_output=True,
                text=True,
                timeout=inp.timeout_seconds,
            )
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            combined = f"{out}\n{err}".strip() if err else out
            if not combined:
                combined = f"(Command completed with exit code {proc.returncode} and no output)"

            return ToolResult(
                ok=(proc.returncode == 0),
                output=combined,
                error=err if proc.returncode != 0 else None,
                metadata={"returncode": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                output="",
                error=f"Command timed out after {inp.timeout_seconds} seconds.",
                metadata={"timeout": True},
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"Command execution error: {e}")


class FindSymbolInput(BaseModel):
    symbol_name: str = Field(..., description="Name of class, function, or method to look up")


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = "Look up symbol definitions, line ranges, signatures, and docstrings."
    input_schema = FindSymbolInput

    def __init__(self, symbol_table: SymbolTable) -> None:
        self.symbol_table = symbol_table

    def execute(self, input: BaseModel) -> ToolResult:
        inp: FindSymbolInput = input  # type: ignore[assignment]
        defs = self.symbol_table.find_definitions(inp.symbol_name)
        if not defs:
            return ToolResult(
                ok=True,
                output=f"No definition found for symbol '{inp.symbol_name}'.",
                metadata={"count": 0},
            )

        lines = [f"Found {len(defs)} definition(s) for '{inp.symbol_name}':"]
        for d in defs:
            parent = f"{d.parent_symbol}." if d.parent_symbol else ""
            sig = f"({d.signature})" if d.signature else ""
            loc = f"{d.file}:{d.line_range[0]}-{d.line_range[1]}"
            lines.append(f"• {d.symbol_type.upper()} {parent}{d.name}{sig} at {loc}")
            if d.docstring:
                lines.append(f"  Docstring: {d.docstring.strip()}")

        return ToolResult(ok=True, output="\n".join(lines), metadata={"count": len(defs)})


class WhereUsedInput(BaseModel):
    symbol_name: str = Field(..., description="Name of the symbol to trace in callers and callees")


class WhereUsedTool(Tool):
    name = "where_used"
    description = (
        "Find where a symbol is referenced, called, and what it calls "
        "using code intelligence and the static call graph."
    )
    input_schema = WhereUsedInput

    def __init__(self, symbol_table: SymbolTable, call_graph: CallGraph | None = None) -> None:
        self.symbol_table = symbol_table
        self.call_graph = call_graph or CallGraph(symbol_table)

    def execute(self, input: BaseModel) -> ToolResult:
        inp: WhereUsedInput = input  # type: ignore[assignment]
        sym = inp.symbol_name

        defs = self.symbol_table.find_definitions(sym)
        refs = self.symbol_table.find_references(sym)
        callers = self.call_graph.get_callers(sym, max_depth=1)
        callees = self.call_graph.get_callees(sym, max_depth=1)

        lines = [f"Usage analysis for symbol '{sym}':"]
        lines.append(f"• Definitions: {len(defs)}")
        for d in defs:
            lines.append(f"  - {d.file}:{d.line_range[0]}-{d.line_range[1]}")

        lines.append(f"• Direct Call Sites / References: {len(refs)}")
        for r in refs:
            lines.append(f"  - Line {r.line_number} in {r.caller_file} ({r.caller_symbol})")

        lines.append(f"• Callers: {', '.join(callers) if callers else 'None'}")
        lines.append(f"• Callees: {', '.join(callees) if callees else 'None'}")

        return ToolResult(
            ok=True,
            output="\n".join(lines),
            metadata={
                "definitions": len(defs),
                "references": len(refs),
                "callers": callers,
                "callees": callees,
            },
        )
