"""Unit tests for Phase 2: Hybrid Retrieval (BM25, score fusion, priority boosts)."""

from typer.testing import CliRunner

from khaslana.cli.main import app
from khaslana.retrieval.bm25 import BM25Index, code_tokenize
from khaslana.retrieval.models import Chunk, SearchResult
from khaslana.retrieval.ranker import HybridRanker

runner = CliRunner()


def test_code_tokenize_camel_and_snake() -> None:
    text = "class AuthService: def get_user_by_id(self, userId): pass"
    tokens = code_tokenize(text)
    assert "authservice" in tokens
    assert "auth" in tokens
    assert "service" in tokens
    assert "get" in tokens
    assert "user" in tokens
    assert "id" in tokens


def test_bm25_index_scoring() -> None:
    chunks = [
        Chunk(
            id="auth.py:1-10",
            text="def authenticate_user(username, password): pass",
            file="auth.py",
            line_range=(1, 10),
            content_hash="h1",
        ),
        Chunk(
            id="calc.py:1-10",
            text="def calculate_sum(a, b): return a + b",
            file="calc.py",
            line_range=(1, 10),
            content_hash="h2",
        ),
    ]

    index = BM25Index(chunks)
    results = index.search("authenticate", top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "auth.py:1-10"
    assert results[0][1] == 1.0


def test_hybrid_ranker_weighted_and_rrf() -> None:
    c1 = Chunk(
        id="c1",
        text="authentication service",
        file="auth.py",
        line_range=(1, 5),
        content_hash="h1",
    )
    c2 = Chunk(
        id="c2",
        text="user model definition",
        file="models.py",
        line_range=(1, 5),
        content_hash="h2",
    )

    all_chunks = {"c1": c1, "c2": c2}

    sem_results = [
        SearchResult(chunk=c1, score=0.9),
        SearchResult(chunk=c2, score=0.4),
    ]
    bm25_scores = {"c1": 0.2, "c2": 0.95}

    ranker = HybridRanker(default_semantic_weight=0.5, default_bm25_weight=0.5)

    # 1. Weighted fusion
    weighted_res = ranker.rank(
        sem_results, bm25_scores, all_chunks, fusion="weighted", top_k=2
    )
    assert len(weighted_res) == 2
    # c2: (0.5 * 0.4) + (0.5 * 0.95) = 0.2 + 0.475 = 0.675
    # c1: (0.5 * 0.9) + (0.5 * 0.2) = 0.45 + 0.1 = 0.55
    assert weighted_res[0].chunk.id == "c2"

    # 2. Focused file boost on c1 (auth.py)
    boosted_res = ranker.rank(
        sem_results,
        bm25_scores,
        all_chunks,
        fusion="weighted",
        focused_file="auth.py",
        top_k=2,
    )
    # c1 base 0.55 * 1.25 = 0.6875 > c2 0.675
    assert boosted_res[0].chunk.id == "c1"
    assert boosted_res[0].boost == 1.25

    # 3. RRF fusion
    rrf_res = ranker.rank(sem_results, bm25_scores, all_chunks, fusion="rrf", top_k=2)
    assert len(rrf_res) == 2
    assert rrf_res[0].metadata["strategy"] == "rrf"


def test_cli_search_missing_index() -> None:
    result = runner.invoke(app, ["search", "auth", "--explain"])
    # Should exit cleanly with a helpful warning message, not crash
    assert result.exit_code in (0, 1)
    assert "index" in result.output.lower() or "warning" in result.output.lower()
