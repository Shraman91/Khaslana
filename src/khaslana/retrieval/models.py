"""Pydantic data models for the RAG and retrieval system."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A semantic chunk of code or text indexed in the system."""

    id: str = Field(
        description="Unique identifier for chunk (e.g. file_path:start_line-end_line)"
    )
    text: str = Field(description="Raw text content of the chunk")
    file: str = Field(description="Repo-relative file path")
    line_range: tuple[int, int] = Field(description="1-based (start_line, end_line)")
    language: str = Field(default="plaintext", description="Programming language or format")
    symbol: str | None = Field(
        default=None, description="Enclosing class/function symbol name if resolved"
    )
    content_hash: str = Field(description="SHA256 hex digest of the file or chunk")
    mtime: float = Field(default=0.0, description="Last modified timestamp")
    callers: list[str] = Field(default_factory=list, description="Known callers of this symbol")
    callees: list[str] = Field(default_factory=list, description="Known callees of this symbol")


class SearchResult(BaseModel):
    """Result of a retrieval search with scoring and metadata."""

    chunk: Chunk
    score: float = Field(description="Overall normalized search score (higher is better)")
    semantic_score: float | None = Field(default=None, description="Vector similarity score")
    bm25_score: float | None = Field(default=None, description="BM25 keyword score")
    boost: float = Field(default=1.0, description="Priority or recency boost multiplier")
    metadata: dict[str, Any] = Field(default_factory=dict)
