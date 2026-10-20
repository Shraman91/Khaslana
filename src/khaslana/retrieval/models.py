"""Pydantic data models for the RAG and retrieval system."""

from __future__ import annotations

from pathlib import Path
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


INDEX_STATE_FILE = ".khaslana/index_state.json"


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hex digest of file contents."""
    import hashlib

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class FileRecord(BaseModel):
    """Metadata tracking a single indexed file."""

    hash: str
    mtime: float
    size: int


class IndexState(BaseModel):
    """State record for incremental indexing and stale detection."""

    root_dir: str
    last_indexed_at: float = 0.0
    files: dict[str, FileRecord] = Field(default_factory=dict)

    def save(self, state_path: Path) -> None:
        """Persist index state to disk as formatted JSON."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, state_path: Path) -> IndexState | None:
        """Load index state from disk if available."""
        import json

        if not state_path.is_file():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except Exception:
            return None


class IncrementalSummary(BaseModel):
    """Summary of an incremental index update."""

    root_dir: str
    updated_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    unchanged_files: int = 0
    elapsed_seconds: float = 0.0

