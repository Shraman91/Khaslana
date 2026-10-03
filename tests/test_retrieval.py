"""Unit tests for retrieval and RAG components (Phase 1)."""

import math
from pathlib import Path

from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import FastHashEmbedding
from khaslana.retrieval.indexer import Indexer
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import ChromaVectorStore, MemoryVectorStore


def test_chunker_line_window() -> None:
    chunker = CodeChunker(target_chunk_tokens=10, max_chunk_tokens=15, overlap_tokens=2)
    # 20 lines, ~1 word per line
    content = "\n".join(f"line {i}" for i in range(20))
    chunks = chunker._chunk_line_window("test.txt", content, "plaintext", "hash1", 0.0)

    assert len(chunks) > 0
    assert chunks[0].file == "test.txt"
    assert chunks[0].content_hash == "hash1"
    # Ensure line ranges make sense
    assert chunks[0].line_range[0] == 1


def test_chunker_python_ast() -> None:
    chunker = CodeChunker(max_chunk_tokens=100)
    content = '''"""Module docstring."""
import os

class MyClass:
    def my_method(self):
        pass

def my_func():
    return 42
'''
    chunks = chunker.chunk_file("test.py", content, "python", 0.0)
    assert len(chunks) >= 3
    symbols = {c.symbol for c in chunks if c.symbol}
    assert "MyClass" in symbols or "MyClass.my_method" in symbols
    assert "my_func" in symbols
    assert "<module_scope>" in symbols


def test_fast_hash_embedding() -> None:
    embedder = FastHashEmbedding(dim=10)
    assert embedder.dimension == 10
    vecs = embedder.encode(["hello world", "test string"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 10

    # Test L2 norm
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_memory_store_integration() -> None:
    chunker = CodeChunker()
    files = {
        "file1.py": "def auth_login():\n    return 'token'",
        "file2.py": "def calc_math():\n    return 42",
    }
    chunks = []
    for fname, content in files.items():
        chunks.extend(chunker.chunk_file(fname, content, "python", 0.0))

    embedder = FastHashEmbedding(dim=16)
    embeddings = embedder.encode([c.text for c in chunks])

    store = MemoryVectorStore()
    store.add_chunks(chunks, embeddings)

    assert store.count() == len(chunks)

    query_emb = embedder.encode_query("auth login token")
    results = store.search(query_emb, top_k=1)
    assert len(results) == 1
    assert "auth_login" in results[0].chunk.text


def test_indexer_workflow(tmp_path: Path) -> None:
    # Setup mock repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def run():\n    pass")
    (repo_dir / "temp.log").write_text("should be ignored if in gitignore")

    (repo_dir / ".gitignore").write_text("*.log\n")

    store = MemoryVectorStore()
    embedder = FastHashEmbedding(dim=16)
    indexer = Indexer(store, embedder)

    summary = indexer.index_directory(repo_dir)
    assert summary.total_files_scanned >= 2
    assert summary.indexed_files == 1
    assert summary.skipped_files >= 1  # log file
    assert store.count() > 0


def test_pipeline_and_citation_formatting() -> None:
    store = MemoryVectorStore()
    embedder = FastHashEmbedding()
    pipeline = RetrievalPipeline(store, embedder)

    # empty result context assembling
    msgs = pipeline.assemble_context_prompt("How do I do X?", [], "You are AI.")
    assert len(msgs) == 2
    assert msgs[1].content == "How do I do X?"

    # Context formatting with citation checks
    chunker = CodeChunker()
    code = "def authenticate_user(username, password):\n    return True"
    chunks = chunker.chunk_file("auth.py", code, "python", 0.0)
    store.add_chunks(chunks, embedder.encode([c.text for c in chunks]))

    results = pipeline.search("authenticate user", top_k=1)
    assert len(results) == 1
    context_str = pipeline.format_search_results(results)
    assert "File: auth.py" in context_str
    assert "Lines: 1-" in context_str
    assert "Symbol: authenticate_user" in context_str


def test_sample_repo_indexing(tmp_path: Path) -> None:
    """Test indexing the sample_repo fixture and querying auth system."""
    sample_repo_path = Path(__file__).parent / "fixtures" / "sample_repo"
    assert sample_repo_path.is_dir()

    chroma_dir = tmp_path / "chroma_test"
    store = ChromaVectorStore(chroma_dir)
    embedder = FastHashEmbedding(dim=64)
    indexer = Indexer(store, embedder)

    summary = indexer.index_directory(sample_repo_path)
    assert summary.indexed_files >= 4
    assert store.count() >= 4

    pipeline = RetrievalPipeline(store, embedder)
    results = pipeline.search("what does the auth system do?", top_k=3)
    assert len(results) > 0

    # Ensure auth.py chunks are discovered
    files_retrieved = {r.chunk.file for r in results}
    assert any("auth.py" in f for f in files_retrieved)
