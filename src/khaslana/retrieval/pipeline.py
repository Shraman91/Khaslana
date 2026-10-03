"""Core retrieval pipeline wrapping indexing, vector search, BM25, and hybrid ranking."""

from __future__ import annotations

from typing import Literal

from khaslana.core.llm import ChatMessage
from khaslana.retrieval.bm25 import BM25Index
from khaslana.retrieval.chunker import count_tokens
from khaslana.retrieval.embeddings import EmbeddingProvider
from khaslana.retrieval.models import Chunk, SearchResult
from khaslana.retrieval.ranker import HybridRanker
from khaslana.retrieval.store import VectorStore


class RetrievalPipeline:
    """High-level hybrid retrieval pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        ranker: HybridRanker | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.store = vector_store
        self.embedder = embedding_provider
        self.ranker = ranker or HybridRanker()
        self.bm25 = bm25_index or BM25Index()

        # Build BM25 index from store if not provided
        all_chunks = self.store.get_all_chunks()
        if all_chunks and not self.bm25.chunks:
            self.bm25.index_chunks(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        fusion: Literal["weighted", "rrf"] | None = None,
        semantic_weight: float | None = None,
        bm25_weight: float | None = None,
        focused_file: str | None = None,
        modified_files: set[str] | None = None,
        mode: Literal["hybrid", "semantic", "bm25"] = "hybrid",
    ) -> list[SearchResult]:
        """Perform search using semantic vector, BM25, or hybrid fusion."""
        all_chunks = self.store.get_all_chunks()
        all_chunks_by_id: dict[str, Chunk] = {c.id: c for c in all_chunks}

        if not all_chunks:
            return []

        # Make sure BM25 index is up to date
        if len(self.bm25.chunks) != len(all_chunks):
            self.bm25.index_chunks(all_chunks)

        if mode == "semantic":
            query_emb = self.embedder.encode_query(query)
            raw_results = self.store.search(query_emb, top_k=top_k)
            for res in raw_results:
                res.boost = self.ranker.compute_boost(
                    res.chunk, focused_file=focused_file, modified_files=modified_files
                )
                res.score = round(res.score * res.boost, 4)
            raw_results.sort(key=lambda x: x.score, reverse=True)
            return raw_results[:top_k]

        if mode == "bm25":
            bm25_pairs = self.bm25.search(query, top_k=top_k)
            results: list[SearchResult] = []
            for chunk, score in bm25_pairs:
                boost = self.ranker.compute_boost(
                    chunk, focused_file=focused_file, modified_files=modified_files
                )
                final_score = round(score * boost, 4)
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=final_score,
                        bm25_score=score,
                        boost=boost,
                    )
                )
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        # Default: Hybrid mode
        query_emb = self.embedder.encode_query(query)
        # Fetch candidate semantic results (over-fetch for better recall fusion)
        candidate_k = max(top_k * 3, 15)
        semantic_results = self.store.search(query_emb, top_k=candidate_k)
        bm25_score_map = self.bm25.get_score_map(query)

        return self.ranker.rank(
            semantic_results=semantic_results,
            bm25_scores=bm25_score_map,
            all_chunks_by_id=all_chunks_by_id,
            fusion=fusion,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            focused_file=focused_file,
            modified_files=modified_files,
            top_k=top_k,
        )

    def format_search_results(self, results: list[SearchResult]) -> str:
        """Format search results into a clean string for LLM inclusion."""
        if not results:
            return ""

        context_blocks = []
        for i, res in enumerate(results, 1):
            chunk = res.chunk
            header_parts = [
                f"File: {chunk.file}",
                f"Lines: {chunk.line_range[0]}-{chunk.line_range[1]}",
            ]
            if chunk.symbol:
                header_parts.append(f"Symbol: {chunk.symbol}")

            header = " | ".join(header_parts)
            block = (
                f"--- [Context Block {i}] ---\n"
                f"{header}\n"
                f"```{(chunk.language or '').lower()}\n"
                f"{chunk.text}\n"
                f"```\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def assemble_context_prompt(
        self,
        query: str,
        results: list[SearchResult],
        base_system_prompt: str,
        max_context_tokens: int = 4000,
    ) -> list[ChatMessage]:
        """Build messages with retrieved context securely tucked into the system prompt."""
        if not results:
            return [
                ChatMessage(role="system", content=base_system_prompt),
                ChatMessage(role="user", content=query),
            ]

        context_str = self.format_search_results(results)

        # Truncate context if it exceeds limits
        while results and count_tokens(context_str) > max_context_tokens:
            results.pop()
            context_str = self.format_search_results(results)

        system_content = (
            f"{base_system_prompt}\n\n"
            "Below is retrieved codebase context that may be relevant to the user's query.\n"
            "When answering, use this context and cite the exact file paths and line ranges.\n"
            "If the answer is not in the context, do not make it up.\n\n"
            f"{context_str}"
        )

        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=query),
        ]
