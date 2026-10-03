"""BM25 keyword search index with code-aware tokenization."""

from __future__ import annotations

import re

from rank_bm25 import BM25Plus  # type: ignore[import-untyped]

from khaslana.retrieval.models import Chunk


def code_tokenize(text: str) -> list[str]:
    """Tokenize source code splitting camelCase, PascalCase, snake_case, and symbols."""
    if not text:
        return []

    # 1. Split on non-alphanumeric characters
    raw_tokens = re.findall(r"[A-Za-z0-9]+", text)
    tokens: list[str] = []

    for tok in raw_tokens:
        tok_lower = tok.lower()
        tokens.append(tok_lower)

        # Split camelCase / PascalCase
        # e.g., 'AuthService' -> ['Auth', 'Service'], 'getUserId' -> ['get', 'User', 'Id']
        sub_tokens = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", tok)
        if len(sub_tokens) > 1:
            for st in sub_tokens:
                st_lower = st.lower()
                if st_lower and st_lower != tok_lower:
                    tokens.append(st_lower)

    return tokens


class BM25Index:
    """In-memory BM25 index over codebase chunks."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Plus | None = None
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build or rebuild the BM25 index over a list of chunks."""
        self.chunks = list(chunks)
        self._corpus_tokens = [code_tokenize(c.text) for c in self.chunks]
        if self._corpus_tokens:
            self._bm25 = BM25Plus(self._corpus_tokens)
        else:
            self._bm25 = None

    def _get_normalized_scores(self, query: str) -> list[float]:
        """Compute normalized [0.0, 1.0] scores for query tokens across chunks."""
        if not self._bm25 or not self.chunks:
            return []

        query_tokens = code_tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0

        if max_score <= 0.0:
            # Fallback to direct term frequency overlap
            tf_scores = [
                sum(doc_tokens.count(qt) for qt in query_tokens)
                for doc_tokens in self._corpus_tokens
            ]
            max_tf = max(tf_scores) if tf_scores else 0
            if max_tf > 0:
                return [float(s / max_tf) for s in tf_scores]
            return [0.0] * len(self.chunks)

        return [float(s / max_score) for s in raw_scores]

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Query the index and return ranked (chunk, normalized_score) pairs."""
        if not self._bm25 or not self.chunks:
            return []

        norm_scores = self._get_normalized_scores(query)
        scored_pairs = [
            (chunk, score)
            for chunk, score in zip(self.chunks, norm_scores, strict=False)
            if score > 0.0
        ]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def get_score_map(self, query: str) -> dict[str, float]:
        """Return a mapping of chunk_id -> normalized BM25 score for all matching chunks."""
        if not self._bm25 or not self.chunks:
            return {}

        norm_scores = self._get_normalized_scores(query)
        return {
            chunk.id: score
            for chunk, score in zip(self.chunks, norm_scores, strict=False)
            if score > 0.0
        }
