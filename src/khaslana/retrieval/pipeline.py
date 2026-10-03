"""Core retrieval pipeline wrapping indexing and search for high-level operations."""

from __future__ import annotations

from khaslana.core.llm import ChatMessage
from khaslana.retrieval.chunker import count_tokens
from khaslana.retrieval.embeddings import EmbeddingProvider
from khaslana.retrieval.models import SearchResult
from khaslana.retrieval.store import VectorStore


class RetrievalPipeline:
    """High-level retrieval pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = vector_store
        self.embedder = embedding_provider

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform semantic search for a query string."""
        query_emb = self.embedder.encode_query(query)
        return self.store.search(query_emb, top_k=top_k)

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

        # Truncate context if it exceeds limits (simple truncation for now)
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
