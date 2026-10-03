"""Pluggable embedding provider interface and implementations."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

from khaslana.core.config import KhaslanaConfig


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension."""
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of text strings into normalized embedding vectors."""
        ...

    def encode_query(self, query: str) -> list[float]:
        """Convenience method to encode a single query string."""
        return self.encode([query])[0]


class FastHashEmbedding(EmbeddingProvider):
    """Fast deterministic pseudo-semantic embedding for instant testing & offline fallback.

    Uses bag of character/word n-grams hashed into a fixed-dimension vector.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            words = text.lower().split()
            if not words:
                vectors.append(vec)
                continue

            # Word tokens and character 3-grams
            tokens: list[str] = list(words)
            for w in words:
                if len(w) >= 3:
                    for i in range(len(w) - 2):
                        tokens.append(w[i : i + 3])

            for token in tokens:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign

            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


class SentenceTransformerEmbedding(EmbeddingProvider):
    """SentenceTransformer embedding using all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._dimension = 384

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension() or 384
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load sentence-transformers model '{self.model_name}': {e}"
                ) from e
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return self._dimension
        return 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        # Encode with normalized embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()  # type: ignore[no-any-return]


def get_embedding_provider(
    config: KhaslanaConfig | None = None,
    force_fast_hash: bool = False,
) -> EmbeddingProvider:
    """Factory function returning the best available embedding provider."""
    if force_fast_hash:
        return FastHashEmbedding()

    model_name = (
        config.retrieval.embedding_model
        if config
        else "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        provider = SentenceTransformerEmbedding(model_name=model_name)
        # Verify model can be loaded
        provider._load_model()
        return provider
    except Exception:
        # Fallback gracefully to FastHashEmbedding
        return FastHashEmbedding()
