"""Pydantic data models for code intelligence, symbols, and call graphs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SymbolType = Literal["function", "method", "class", "variable", "interface", "module"]


class Symbol(BaseModel):
    """A defined code symbol (function, class, method, etc.)."""

    name: str = Field(description="Symbol identifier name")
    symbol_type: SymbolType = Field(description="Type classification of symbol")
    file: str = Field(description="Repository-relative file path")
    line_range: tuple[int, int] = Field(description="1-based (start_line, end_line)")
    docstring: str | None = Field(default=None, description="Extracted docstring if present")
    parent_symbol: str | None = Field(
        default=None, description="Enclosing class or parent symbol name"
    )
    signature: str | None = Field(default=None, description="Parameter signature if applicable")


class ImportEdge(BaseModel):
    """An import dependency between modules."""

    file: str = Field(description="File containing the import statement")
    imported_module: str = Field(description="Module name being imported")
    imported_symbol: str | None = Field(default=None, description="Specific symbol imported")
    alias: str | None = Field(default=None, description="Alias used if imported as")
    line_number: int = Field(default=1, description="Line number of import statement")


class CallEdge(BaseModel):
    """A call relationship between code symbols."""

    caller_file: str = Field(description="File where the call occurs")
    caller_symbol: str = Field(description="Enclosing symbol making the call")
    callee_name: str = Field(description="Name of symbol/function being called")
    line_number: int = Field(description="Line number of call site")


class ParsedFile(BaseModel):
    """Complete structural analysis of a source file."""

    file: str = Field(description="Repository-relative file path")
    language: str = Field(default="python", description="Programming language")
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[ImportEdge] = Field(default_factory=list)
    calls: list[CallEdge] = Field(default_factory=list)
