"""Vector store abstractions and implementations (ChromaDB + Memory fallback)."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from khaslana.retrieval.models import Chunk, SearchResult


class VectorStore(ABC):
    """Abstract interface for storing and querying chunk embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add or update chunks with their vector embeddings."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for the most similar chunks by cosine similarity."""
        ...

    @abstractmethod
    def delete_file(self, file_path: str) -> None:
        """Delete all chunks belonging to a specific file."""
        ...

    @abstractmethod
    def get_all_chunks(self) -> list[Chunk]:
        """Retrieve all indexed chunks."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of indexed chunks."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear the entire index."""
        ...


class MemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity (for fast testing & fallbacks)."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, emb in zip(chunks, embeddings, strict=False):
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = emb

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if not self._chunks:
            return []

        scored: list[tuple[float, Chunk]] = []
        for c_id, chunk in self._chunks.items():
            emb = self._embeddings[c_id]
            sim = self._cosine_similarity(query_embedding, emb)
            # Map [-1, 1] to [0, 1]
            norm_score = max(0.0, (sim + 1.0) / 2.0)
            scored.append((norm_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for score, chunk in scored[:top_k]:
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    semantic_score=score,
                )
            )
        return results

    def delete_file(self, file_path: str) -> None:
        to_del = [c_id for c_id, chunk in self._chunks.items() if chunk.file == file_path]
        for c_id in to_del:
            self._chunks.pop(c_id, None)
            self._embeddings.pop(c_id, None)

    def get_all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = {}
        self._embeddings = {}


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB vector store."""

    def __init__(self, persist_dir: Path | str, collection_name: str = "khaslana_code") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = []

        for c in chunks:
            metadatas.append(
                {
                    "file": c.file,
                    "start_line": c.line_range[0],
                    "end_line": c.line_range[1],
                    "language": c.language,
                    "symbol": c.symbol or "",
                    "content_hash": c.content_hash,
                    "mtime": c.mtime,
                    "callers": json.dumps(c.callers),
                    "callees": json.dumps(c.callees),
                }
            )

        # Batch upsert into Chroma
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        count = self.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        if not res or not res["ids"] or not res["ids"][0]:
            return results

        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        for c_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            # Chroma returns cosine distance (0 to 2 for cosine). Similarity = 1 - (dist / 2)
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            raw_callers = meta.get("callers", "[]")
            raw_callees = meta.get("callees", "[]")
            callers = json.loads(raw_callers) if isinstance(raw_callers, str) else []
            callees = json.loads(raw_callees) if isinstance(raw_callees, str) else []

            chunk = Chunk(
                id=c_id,
                text=doc,
                file=meta["file"],
                line_range=(int(meta["start_line"]), int(meta["end_line"])),
                language=meta.get("language", "plaintext"),
                symbol=meta.get("symbol") or None,
                content_hash=meta.get("content_hash", ""),
                mtime=float(meta.get("mtime", 0.0)),
                callers=callers,
                callees=callees,
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=similarity,
                    semantic_score=similarity,
                )
            )

        return results

    def delete_file(self, file_path: str) -> None:
        self._collection.delete(where={"file": file_path})

    def get_all_chunks(self) -> list[Chunk]:
        data = self._collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        if not data or not data["ids"]:
            return chunks

        for c_id, doc, meta in zip(
            data["ids"], data["documents"], data["metadatas"], strict=False
        ):
            raw_callers = meta.get("callers", "[]")
            raw_callees = meta.get("callees", "[]")
            callers = json.loads(raw_callers) if isinstance(raw_callers, str) else []
            callees = json.loads(raw_callees) if isinstance(raw_callees, str) else []
            chunks.append(
                Chunk(
                    id=c_id,
                    text=doc,
                    file=meta["file"],
                    line_range=(int(meta["start_line"]), int(meta["end_line"])),
                    language=meta.get("language", "plaintext"),
                    symbol=meta.get("symbol") or None,
                    content_hash=meta.get("content_hash", ""),
                    mtime=float(meta.get("mtime", 0.0)),
                    callers=callers,
                    callees=callees,
                )
            )
        return chunks

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
