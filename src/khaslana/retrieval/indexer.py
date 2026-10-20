"""Directory indexer honoring gitignore, filtering binaries, chunking, and embedding."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pathspec
from pydantic import BaseModel

from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.parser import parse_source
from khaslana.intelligence.symbols import SymbolTable
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import EmbeddingProvider
from khaslana.retrieval.models import (
    INDEX_STATE_FILE,
    Chunk,
    FileRecord,
    IndexState,
    compute_file_hash,
)
from khaslana.retrieval.store import VectorStore

DEFAULT_IGNORES = [
    ".git",
    ".git/*",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".DS_Store",
    ".env",
    ".env.*",
    "node_modules",
    "node_modules/*",
    "venv",
    "venv/*",
    ".venv",
    ".venv/*",
    "__pycache__",
    "__pycache__/*",
    "dist",
    "dist/*",
    "build",
    "build/*",
    ".khaslana",
    ".khaslana/*",
    ".idea",
    ".vscode",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.egg-info",
    "*.egg-info/*",
    ".tox",
    ".coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "package-lock.json",
    "poetry.lock",
    "yarn.lock",
    "Cargo.lock",
]

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
    ".dockerfile": "dockerfile",
}


class IndexSummary(BaseModel):
    """Summary of an indexing run."""

    root_dir: str
    total_files_scanned: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    elapsed_seconds: float = 0.0


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by looking for null bytes in initial chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        return True


def load_gitignore_spec(root_dir: Path) -> pathspec.PathSpec:
    """Load .gitignore from project root combined with DEFAULT_IGNORES."""
    patterns = list(DEFAULT_IGNORES)
    gitignore_path = root_dir / ".gitignore"
    if gitignore_path.is_file():
        try:
            with open(gitignore_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except Exception:
            pass
    return pathspec.PathSpec.from_lines("gitignore", patterns)


class Indexer:
    """Indexes source code repositories into vector stores and symbol tables."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        chunker: CodeChunker | None = None,
        symbol_table: SymbolTable | None = None,
        max_file_size_bytes: int = 1_000_000,  # 1MB
    ) -> None:
        self.store = vector_store
        self.embedder = embedding_provider
        self.chunker = chunker or CodeChunker()
        self.symbol_table = symbol_table
        self.max_file_size_bytes = max_file_size_bytes

    def index_directory(
        self,
        root_dir: Path | str,
        incremental: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexSummary:
        """Walk directory, chunk, embed, and store files."""
        start_time = time.time()
        root = Path(root_dir).resolve()
        spec = load_gitignore_spec(root)

        summary = IndexSummary(root_dir=str(root))
        files_to_process: list[Path] = []

        # 1. Discover all matching files
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)

            # Filter out ignored directories in place
            dirnames[:] = [
                d
                for d in dirnames
                if not spec.match_file(str((rel_dir / d).as_posix()))
                and not spec.match_file(f"{d}/")
                and d not in {".git", "node_modules", "venv", ".venv", "__pycache__", ".khaslana"}
            ]

            for fname in filenames:
                summary.total_files_scanned += 1
                rel_path = (rel_dir / fname).as_posix()
                full_path = Path(dirpath) / fname

                # Check gitignore
                if spec.match_file(rel_path) or spec.match_file(fname):
                    summary.skipped_files += 1
                    continue

                # Check file size
                try:
                    size = full_path.stat().st_size
                    if size > self.max_file_size_bytes or size == 0:
                        summary.skipped_files += 1
                        continue
                except Exception:
                    summary.skipped_files += 1
                    continue

                # Check if binary
                if is_binary_file(full_path):
                    summary.skipped_files += 1
                    continue

                files_to_process.append(full_path)

        total_files = len(files_to_process)
        all_chunks: list[Chunk] = []
        indexed_records: dict[str, FileRecord] = {}

        # 2. Chunk files and index symbols
        for i, file_path in enumerate(files_to_process):
            rel_path = file_path.relative_to(root).as_posix()
            if progress_callback:
                progress_callback(rel_path, i + 1, total_files)

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                stat = file_path.stat()
                mtime = stat.st_mtime
                ext = file_path.suffix.lower()
                lang = EXTENSION_TO_LANGUAGE.get(ext, "plaintext")

                # Parse into symbol table if available
                if self.symbol_table:
                    parsed = parse_source(rel_path, content, language=lang)
                    self.symbol_table.index_parsed_file(parsed)

                chunks = self.chunker.chunk_file(
                    file_path=rel_path,
                    content=content,
                    language=lang,
                    mtime=mtime,
                )
                if chunks:
                    all_chunks.extend(chunks)
                    summary.indexed_files += 1
                    indexed_records[rel_path] = FileRecord(
                        hash=compute_file_hash(file_path),
                        mtime=mtime,
                        size=stat.st_size,
                    )
                else:
                    summary.skipped_files += 1
            except Exception:
                summary.skipped_files += 1

        # 3. Enrich chunks with call graph if symbol table is active
        if self.symbol_table and all_chunks:
            call_graph = CallGraph(self.symbol_table)
            for chunk in all_chunks:
                if chunk.symbol and chunk.symbol != "<module_scope>":
                    chunk.callers = call_graph.get_callers(chunk.symbol, max_depth=1)
                    chunk.callees = call_graph.get_callees(chunk.symbol, max_depth=1)

        summary.total_chunks = len(all_chunks)

        # 4. Batch embed and store chunks
        if all_chunks:
            batch_size = 64
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                texts = [c.text for c in batch]
                embeddings = self.embedder.encode(texts)
                self.store.add_chunks(batch, embeddings)

        # 5. Persist index state
        try:
            state = IndexState(
                root_dir=str(root),
                last_indexed_at=time.time(),
                files=indexed_records,
            )
            state.save(root / INDEX_STATE_FILE)
        except Exception:
            pass

        summary.elapsed_seconds = round(time.time() - start_time, 2)
        return summary

