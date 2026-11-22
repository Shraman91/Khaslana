"""Tests for packaging, optional extras, and lazy imports."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from khaslana.core.errors import KhaslanaError, SandboxUnavailableError
from khaslana.retrieval.embeddings import SentenceTransformerEmbedding
from khaslana.retrieval.store import ChromaVectorStore
from khaslana.sandbox.docker_runner import DockerRunner


def test_missing_chromadb_raises_clear_error(tmp_path: Path) -> None:
    with patch.dict(sys.modules, {"chromadb": None, "chromadb.config": None}):
        with pytest.raises(KhaslanaError) as exc_info:
            ChromaVectorStore(persist_dir=tmp_path)
        assert "ChromaDB is not installed" in exc_info.value.message
        assert "khaslana[embeddings]" in (exc_info.value.hint or "")


def test_missing_sentence_transformers_raises_clear_error() -> None:
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        embedder = SentenceTransformerEmbedding()
        with pytest.raises(KhaslanaError) as exc_info:
            embedder.encode(["test query"])
        assert "sentence-transformers is not installed" in exc_info.value.message
        assert "khaslana[embeddings]" in (exc_info.value.hint or "")


def test_missing_docker_raises_clear_error() -> None:
    with patch.dict(sys.modules, {"docker": None}):
        runner = DockerRunner()
        with pytest.raises(SandboxUnavailableError) as exc_info:
            runner._get_client()
        assert "Docker Python SDK is not installed" in exc_info.value.message
        assert "khaslana[sandbox]" in (exc_info.value.hint or "")
