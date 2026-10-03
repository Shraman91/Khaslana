"""Structure-aware and line-window chunker for code and text files."""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any

from khaslana.retrieval.models import Chunk

# Try importing tiktoken with fallback estimator
try:
    import tiktoken

    _TIKTOKEN_ENCODER: Any = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


def count_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken or fallback estimator."""
    if not text:
        return 0
    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Fallback heuristic: words * 1.33 + punctuation
    words = len(re.findall(r"\w+|[^\w\s]", text))
    return max(1, int(words * 1.1))


def _hash_content(text: str) -> str:
    """Generate SHA256 hex digest for content."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class CodeChunker:
    """Structure-aware chunker that splits code by AST (Python) and line windows."""

    def __init__(
        self,
        target_chunk_tokens: int = 400,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 50,
    ) -> None:
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(
        self,
        file_path: str,
        content: str,
        language: str = "plaintext",
        mtime: float = 0.0,
    ) -> list[Chunk]:
        """Chunk a file into a list of Chunk objects."""
        if not content.strip():
            return []

        file_hash = _hash_content(content)

        if language == "python" or file_path.endswith(".py"):
            try:
                chunks = self._chunk_python_ast(file_path, content, file_hash, mtime)
                if chunks:
                    return chunks
            except Exception:
                # Fall back to line window chunking if AST parsing fails
                pass

        return self._chunk_line_window(file_path, content, language, file_hash, mtime)

    def _chunk_python_ast(
        self,
        file_path: str,
        content: str,
        file_hash: str,
        mtime: float,
    ) -> list[Chunk]:
        """Split Python code into class and function units using AST."""
        tree = ast.parse(content, filename=file_path)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        units: list[dict[str, Any]] = []

        # Find all top-level functions and classes
        covered_line_ranges: list[tuple[int, int]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                symbol_name = node.name
                covered_line_ranges.append((start, end))

                # If class, also inspect methods for nested units if class is large
                if isinstance(node, ast.ClassDef):
                    class_text = "".join(lines[start - 1 : end])
                    if count_tokens(class_text) > self.max_chunk_tokens:
                        # Extract methods individually
                        for sub_node in node.body:
                            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                m_start = getattr(sub_node, "lineno", start)
                                m_end = getattr(sub_node, "end_lineno", m_start)
                                m_text = "".join(lines[m_start - 1 : m_end])
                                units.append(
                                    {
                                        "start": m_start,
                                        "end": m_end,
                                        "text": m_text,
                                        "symbol": f"{symbol_name}.{sub_node.name}",
                                    }
                                )
                        continue

                unit_text = "".join(lines[start - 1 : end])
                units.append(
                    {
                        "start": start,
                        "end": end,
                        "text": unit_text,
                        "symbol": symbol_name,
                    }
                )

        # Collect uncovered header / top-level code (e.g. imports, constants)
        # Find gaps between covered line ranges
        covered_line_ranges.sort(key=lambda r: r[0])
        curr_line = 1
        for start, end in covered_line_ranges:
            if start > curr_line:
                gap_text = "".join(lines[curr_line - 1 : start - 1]).strip()
                if gap_text:
                    units.append(
                        {
                            "start": curr_line,
                            "end": start - 1,
                            "text": "".join(lines[curr_line - 1 : start - 1]),
                            "symbol": "<module_scope>",
                        }
                    )
            curr_line = max(curr_line, end + 1)

        if curr_line <= total_lines:
            trailing = "".join(lines[curr_line - 1 : total_lines]).strip()
            if trailing:
                units.append(
                    {
                        "start": curr_line,
                        "end": total_lines,
                        "text": "".join(lines[curr_line - 1 : total_lines]),
                        "symbol": "<module_scope>",
                    }
                )

        units.sort(key=lambda u: u["start"])

        # Split any oversized units into sub-chunks
        final_chunks: list[Chunk] = []
        for u in units:
            text = u["text"]
            start_line = u["start"]
            symbol = u["symbol"]
            tokens = count_tokens(text)

            if tokens <= self.max_chunk_tokens:
                c_id = f"{file_path}:{start_line}-{u['end']}"
                final_chunks.append(
                    Chunk(
                        id=c_id,
                        text=text.strip(),
                        file=file_path,
                        line_range=(start_line, u["end"]),
                        language="python",
                        symbol=symbol,
                        content_hash=file_hash,
                        mtime=mtime,
                    )
                )
            else:
                # Sub-chunk with line window
                sub_chunks = self._chunk_line_window(
                    file_path=file_path,
                    content=text,
                    language="python",
                    file_hash=file_hash,
                    mtime=mtime,
                    base_line_offset=start_line - 1,
                    symbol=symbol,
                )
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _chunk_line_window(
        self,
        file_path: str,
        content: str,
        language: str,
        file_hash: str,
        mtime: float,
        base_line_offset: int = 0,
        symbol: str | None = None,
    ) -> list[Chunk]:
        """Split content into overlapping windows of lines respecting token budgets."""
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[Chunk] = []
        n_lines = len(lines)
        start_idx = 0

        while start_idx < n_lines:
            end_idx = start_idx
            curr_tokens = 0

            # Accumulate lines up to target_chunk_tokens
            while end_idx < n_lines:
                line_tokens = count_tokens(lines[end_idx])
                if curr_tokens + line_tokens > self.max_chunk_tokens and end_idx > start_idx:
                    break
                curr_tokens += line_tokens
                end_idx += 1

            chunk_lines = lines[start_idx:end_idx]
            chunk_text = "".join(chunk_lines).strip()

            if chunk_text:
                start_line_no = base_line_offset + start_idx + 1
                end_line_no = base_line_offset + end_idx
                chunk_id = f"{file_path}:{start_line_no}-{end_line_no}"

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        file=file_path,
                        line_range=(start_line_no, end_line_no),
                        language=language,
                        symbol=symbol,
                        content_hash=file_hash,
                        mtime=mtime,
                    )
                )

            if end_idx >= n_lines:
                break

            # Calculate step forward with overlap
            # Roll back lines until we roughly meet overlap_tokens
            overlap_count = 0
            step_back = 0
            for idx in range(end_idx - 1, start_idx, -1):
                t = count_tokens(lines[idx])
                if overlap_count + t > self.overlap_tokens:
                    break
                overlap_count += t
                step_back += 1

            start_idx = max(start_idx + 1, end_idx - step_back)

        return chunks
