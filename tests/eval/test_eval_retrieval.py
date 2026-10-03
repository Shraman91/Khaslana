"""Evaluation harness benchmarking Recall@K across Semantic, BM25, and Hybrid retrieval."""

from pathlib import Path

from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import FastHashEmbedding
from khaslana.retrieval.indexer import Indexer
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import MemoryVectorStore

EVAL_BENCHMARK_DATA: list[dict[str, str]] = [
    {"query": "authenticate user verify password credentials", "expected_file": "auth.py"},
    {"query": "SessionToken expires_at lifetime check", "expected_file": "models.py"},
    {"query": "find_user_by_username in database memory", "expected_file": "database.py"},
    {"query": "post_login endpoint router dispatch", "expected_file": "api.py"},
    {"query": "format_json_response pretty print json", "expected_file": "utils.py"},
    {"query": "hash_password SHA256 secret salt", "expected_file": "auth.py"},
    {"query": "list_products inventory catalog", "expected_file": "database.py"},
    {"query": "validate_token active session check", "expected_file": "auth.py"},
]


def calculate_recall_at_k(
    pipeline: RetrievalPipeline,
    queries: list[dict[str, str]],
    mode: str,
    k: int,
    fusion: str = "weighted",
) -> float:
    """Calculate Recall@K metric."""
    hits = 0
    for item in queries:
        q = item["query"]
        expected = item["expected_file"]
        # Use literal mode
        if mode == "semantic":
            results = pipeline.search(q, top_k=k, mode="semantic")
        elif mode == "bm25":
            results = pipeline.search(q, top_k=k, mode="bm25")
        else:
            results = pipeline.search(
                q, top_k=k, mode="hybrid", fusion="rrf" if fusion == "rrf" else "weighted"
            )

        retrieved_files = [r.chunk.file.replace("\\", "/") for r in results]
        if any(expected in rf for rf in retrieved_files):
            hits += 1

    return round(hits / len(queries), 4)


def test_retrieval_evaluation_benchmark() -> None:
    """Run retrieval evaluation harness and print benchmark comparison table."""
    sample_repo_path = Path(__file__).parent.parent / "fixtures" / "sample_repo"
    assert sample_repo_path.is_dir()

    store = MemoryVectorStore()
    embedder = FastHashEmbedding(dim=128)
    chunker = CodeChunker()
    indexer = Indexer(store, embedder, chunker)
    indexer.index_directory(sample_repo_path)

    pipeline = RetrievalPipeline(store, embedder)

    k_values = [1, 3, 5]
    modes = [
        ("Semantic Only", "semantic", "weighted"),
        ("BM25 Only", "bm25", "weighted"),
        ("Hybrid (Weighted)", "hybrid", "weighted"),
        ("Hybrid (RRF)", "hybrid", "rrf"),
    ]

    print("\n\n=== Retrieval Evaluation Benchmark (Recall@K) ===")
    print("| Method              | Recall@1 | Recall@3 | Recall@5 |")
    print("|---------------------|----------|----------|----------|")

    results_table: dict[str, dict[int, float]] = {}

    for label, mode_str, fusion_str in modes:
        results_table[label] = {}
        for k in k_values:
            score = calculate_recall_at_k(
                pipeline, EVAL_BENCHMARK_DATA, mode_str, k, fusion=fusion_str
            )
            results_table[label][k] = score

        print(
            f"| {label:<19} | {results_table[label][1]:<8.2f} | "
            f"{results_table[label][3]:<8.2f} | {results_table[label][5]:<8.2f} |"
        )

    # Hybrid should achieve top recall on ground truth queries
    assert results_table["Hybrid (Weighted)"][3] >= 0.85
    assert results_table["Hybrid (Weighted)"][5] >= 0.85
