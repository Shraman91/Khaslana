"""Unit tests for the retrieval evaluation harness and benchmark command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from khaslana.cli.eval_cmd import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_pipeline_on_dataset,
    is_chunk_relevant,
)
from khaslana.cli.main import app
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import FastHashEmbedding
from khaslana.retrieval.indexer import Indexer
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import MemoryVectorStore

runner = CliRunner()
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_metric_calculations() -> None:
    # MRR: rank 1 -> 1.0, rank 2 -> 0.5, None -> 0.0 => average 0.5
    ranks: list[int | None] = [1, 2, None]
    assert round(compute_mrr(ranks), 4) == 0.5

    # Empty ranks
    assert compute_mrr([]) == 0.0

    # Recall@1: 1 hit in top 1 out of 3 queries => 0.3333
    assert round(compute_recall_at_k(ranks, 1), 2) == 0.33
    # Recall@3: 2 hits in top 3 out of 3 queries => 0.67
    assert round(compute_recall_at_k(ranks, 3), 2) == 0.67

    # nDCG@K: Perfect hit at rank 1
    ndcg_perfect = compute_ndcg_at_k([1, 0, 0], num_ideal_relevant=1, k=10)
    assert round(ndcg_perfect, 4) == 1.0

    # Hit at rank 2
    ndcg_rank2 = compute_ndcg_at_k([0, 1, 0], num_ideal_relevant=1, k=10)
    assert 0.0 < ndcg_rank2 < 1.0

    # No hits
    assert compute_ndcg_at_k([0, 0, 0], num_ideal_relevant=1, k=10) == 0.0


def test_is_chunk_relevant() -> None:
    assert is_chunk_relevant(
        chunk_file="src/click/core.py",
        chunk_symbol="Context",
        expected_files=["src/click/core.py"],
        expected_symbols=["Context"],
    )

    # Windows vs Unix path separators
    assert is_chunk_relevant(
        chunk_file="src\\click\\core.py",
        chunk_symbol=None,
        expected_files=["src/click/core.py"],
    )

    # Substring file match
    assert is_chunk_relevant(
        chunk_file="/full/path/to/src/click/parser.py",
        chunk_symbol=None,
        expected_files=["src/click/parser.py"],
    )

    # Irrelevant chunk
    assert not is_chunk_relevant(
        chunk_file="src/click/other.py",
        chunk_symbol="Other",
        expected_files=["src/click/core.py"],
        expected_symbols=["Context"],
    )


def test_evaluate_pipeline_on_dataset() -> None:
    store = MemoryVectorStore()
    embedder = FastHashEmbedding(dim=384)
    chunker = CodeChunker()
    indexer = Indexer(
        vector_store=store,
        embedding_provider=embedder,
        chunker=chunker,
    )
    indexer.index_directory(SAMPLE_REPO)

    pipeline = RetrievalPipeline(vector_store=store, embedding_provider=embedder)

    queries = [
        {"query": "database connection pool", "expected_files": ["database.py"]},
        {"query": "password hashing and authentication", "expected_files": ["auth.py"]},
    ]

    modes = ["semantic", "bm25", "hybrid-weighted", "hybrid-rrf"]
    results = evaluate_pipeline_on_dataset(pipeline, queries, modes=modes, top_k=5)

    assert set(results.keys()) == set(modes)
    for mode in modes:
        res = results[mode]
        assert res.query_count == 2
        assert 0.0 <= res.recall_1 <= 1.0
        assert 0.0 <= res.recall_5 <= 1.0
        assert 0.0 <= res.mrr <= 1.0
        assert 0.0 <= res.ndcg_10 <= 1.0


def test_cli_eval_command(tmp_path: Path) -> None:
    # Create a local test dataset pointing to the sample_repo
    sample_dataset = {
        "name": "sample_repo",
        "language": "python",
        "queries": [
            {"query": "database query", "expected_files": ["database.py"]},
            {"query": "authenticate token", "expected_files": ["auth.py"]},
        ],
    }
    dataset_file = tmp_path / "sample_dataset.json"
    dataset_file.write_text(json.dumps(sample_dataset), encoding="utf-8")

    out_file = tmp_path / "results.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--dataset",
            str(dataset_file),
            "--cache-dir",
            str(SAMPLE_REPO.parent),
            "--modes",
            "bm25,hybrid-rrf",
            "--output",
            str(out_file),
        ],
    )

    assert result.exit_code == 0
    assert "Retrieval Benchmark: sample_repo" in result.output
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert "sample_repo" in saved
    assert saved["sample_repo"]["status"] == "completed"


def test_cli_eval_unavailable_repo_graceful(tmp_path: Path) -> None:
    # Non-existent repository URL and cache dir
    dummy_dataset = {
        "name": "nonexistent_repo_xyz",
        "language": "python",
        "repo_url": "https://invalid.example.com/nonexistent.git",
        "commit": "deadbeef",
        "queries": [{"query": "test", "expected_files": ["test.py"]}],
    }
    dataset_file = tmp_path / "dummy.json"
    dataset_file.write_text(json.dumps(dummy_dataset), encoding="utf-8")

    result = runner.invoke(
        app,
        ["eval", "--dataset", str(dataset_file), "--cache-dir", str(tmp_path / "cache")],
    )

    assert result.exit_code == 0
    assert "not yet run" in result.output
