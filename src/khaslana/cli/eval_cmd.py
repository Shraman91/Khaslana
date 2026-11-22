"""Retrieval evaluation harness and benchmark command for Khaslana.

Evaluates information retrieval performance across Semantic, BM25, Hybrid-Weighted,
and Hybrid-RRF ranking strategies using Recall@K, MRR, and nDCG@10.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import FastHashEmbedding
from khaslana.retrieval.indexer import Indexer
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import MemoryVectorStore

console = Console()
DEFAULT_CACHE_DIR = Path.home() / ".khaslana" / "cache" / "eval_repos"


class EvalMetricResult(BaseModel):
    """Aggregated evaluation metrics for a retrieval mode."""

    mode: str = Field(description="Retrieval strategy name")
    query_count: int = Field(description="Total queries evaluated")
    recall_1: float = Field(description="Proportion of queries with hit at rank 1")
    recall_3: float = Field(description="Proportion of queries with hit at rank <= 3")
    recall_5: float = Field(description="Proportion of queries with hit at rank <= 5")
    recall_10: float = Field(description="Proportion of queries with hit at rank <= 10")
    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg_10: float = Field(description="Normalized Discounted Cumulative Gain at rank 10")


def compute_mrr(ranks: list[int | None]) -> float:
    """Compute Mean Reciprocal Rank (MRR) from 1-based ranks (None if not found)."""
    if not ranks:
        return 0.0
    reciprocals = [(1.0 / r) if (r is not None and r > 0) else 0.0 for r in ranks]
    return sum(reciprocals) / len(reciprocals)


def compute_recall_at_k(ranks: list[int | None], k: int) -> float:
    """Compute Recall@K: proportion of queries with a hit at rank <= k."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if (r is not None and 1 <= r <= k))
    return hits / len(ranks)


def compute_ndcg_at_k(hits_binary: list[int], num_ideal_relevant: int = 1, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (nDCG@K)."""
    if not hits_binary or num_ideal_relevant <= 0:
        return 0.0

    # Actual DCG@K
    dcg = 0.0
    for rank_idx, hit in enumerate(hits_binary[:k], 1):
        if hit:
            dcg += 1.0 / math.log2(rank_idx + 1)

    # Ideal DCG@K
    ideal_hits = min(num_ideal_relevant, k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0

    return min(1.0, dcg / idcg)


def is_chunk_relevant(
    chunk_file: str,
    chunk_symbol: str | None,
    expected_files: list[str],
    expected_symbols: list[str] | None = None,
) -> bool:
    """Check if retrieved chunk matches expected file paths or symbol names."""
    normalized_chunk = chunk_file.replace("\\", "/").lower()
    for expected in expected_files:
        normalized_exp = expected.replace("\\", "/").lower()
        if (
            normalized_chunk == normalized_exp
            or normalized_chunk.endswith(normalized_exp)
            or normalized_exp.endswith(normalized_chunk)
        ):
            return True

    if expected_symbols and chunk_symbol:
        chunk_sym_lower = chunk_symbol.lower()
        for exp_sym in expected_symbols:
            if exp_sym.lower() in chunk_sym_lower:
                return True

    return False


def evaluate_pipeline_on_dataset(
    pipeline: RetrievalPipeline,
    queries: list[dict[str, Any]],
    modes: list[str],
    top_k: int = 10,
) -> dict[str, EvalMetricResult]:
    """Evaluate a configured RetrievalPipeline on query ground-truth pairs."""
    results_by_mode: dict[str, EvalMetricResult] = {}

    for mode in modes:
        ranks: list[int | None] = []
        ndcg_scores: list[float] = []

        for q_item in queries:
            query = q_item["query"]
            expected_files = q_item.get("expected_files", [])
            expected_symbols = q_item.get("expected_symbols", [])

            # Execute search based on mode
            if mode == "semantic":
                search_results = pipeline.search(query, top_k=top_k, mode="semantic")
            elif mode == "bm25":
                search_results = pipeline.search(query, top_k=top_k, mode="bm25")
            elif mode == "hybrid-rrf":
                search_results = pipeline.search(query, top_k=top_k, mode="hybrid", fusion="rrf")
            else:  # hybrid-weighted
                search_results = pipeline.search(
                    query, top_k=top_k, mode="hybrid", fusion="weighted"
                )

            hits_binary: list[int] = []
            first_hit_rank: int | None = None

            for rank_1b, res in enumerate(search_results, 1):
                is_hit = is_chunk_relevant(
                    res.chunk.file,
                    res.chunk.symbol,
                    expected_files,
                    expected_symbols,
                )
                hits_binary.append(1 if is_hit else 0)
                if is_hit and first_hit_rank is None:
                    first_hit_rank = rank_1b

            ranks.append(first_hit_rank)
            ideal_count = max(1, len(expected_files) + len(expected_symbols))
            ndcg_scores.append(
                compute_ndcg_at_k(hits_binary, num_ideal_relevant=ideal_count, k=top_k)
            )

        q_count = len(queries)
        recall_1 = compute_recall_at_k(ranks, 1)
        recall_3 = compute_recall_at_k(ranks, 3)
        recall_5 = compute_recall_at_k(ranks, 5)
        recall_10 = compute_recall_at_k(ranks, 10)
        mrr = compute_mrr(ranks)
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

        results_by_mode[mode] = EvalMetricResult(
            mode=mode,
            query_count=q_count,
            recall_1=round(recall_1, 4),
            recall_3=round(recall_3, 4),
            recall_5=round(recall_5, 4),
            recall_10=round(recall_10, 4),
            mrr=round(mrr, 4),
            ndcg_10=round(avg_ndcg, 4),
        )

    return results_by_mode


def ensure_eval_repo(
    repo_url: str,
    commit: str,
    target_dir: Path,
) -> bool:
    """Clone and checkout pinned repository to cache directory."""
    if (target_dir / ".git").exists():
        return True

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "50", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(target_dir), "checkout", commit],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True
    except Exception:
        return False


def eval_cmd(
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to evaluation dataset JSON file (defaults to benchmarks/)",
        ),
    ] = None,
    modes: Annotated[
        str,
        typer.Option(
            "--modes",
            "-m",
            help="Comma-separated retrieval modes (semantic,bm25,hybrid-weighted,hybrid-rrf)",
        ),
    ] = "semantic,bm25,hybrid-weighted,hybrid-rrf",
    cache_dir: Annotated[
        Path,
        typer.Option(
            "--cache-dir",
            help="Cache directory for cloned benchmark repositories",
        ),
    ] = DEFAULT_CACHE_DIR,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit number of evaluated queries"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON file path to store benchmark results"),
    ] = None,
) -> None:
    """Run retrieval evaluation benchmark comparing Semantic, BM25, and Hybrid fusion."""
    eval_modes = [m.strip().lower() for m in modes.split(",") if m.strip()]
    dataset_files: list[Path] = []

    if dataset:
        dataset_files = [dataset]
    else:
        bench_dir = Path("benchmarks")
        if bench_dir.exists():
            dataset_files = sorted(bench_dir.glob("*_repo.json"))

    if not dataset_files:
        console.print(
            "[yellow]No benchmark dataset specifications found in 'benchmarks/'.[/yellow]"
        )
        raise typer.Exit(code=1)

    all_results: dict[str, dict[str, Any]] = {}

    for d_path in dataset_files:
        try:
            data = json.loads(d_path.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[red]Error reading dataset {d_path}: {e}[/red]")
            continue

        repo_name = data.get("name", d_path.stem)
        repo_url = data.get("repo_url")
        commit = data.get("commit", "HEAD")
        queries = data.get("queries", [])
        if limit and limit > 0:
            queries = queries[:limit]

        console.print(
            f"\n[bold blue]=== Evaluating Dataset:[/bold blue] {repo_name} ({len(queries)} queries)"
        )

        target_repo_dir = cache_dir / repo_name
        repo_ready = False

        if target_repo_dir.exists():
            repo_ready = True
        elif repo_url:
            short_sha = commit[:7] if commit else ""
            console.print(f"[dim]Cloning pinned repo {repo_url} (commit {short_sha})...[/dim]")
            repo_ready = ensure_eval_repo(repo_url, commit, target_repo_dir)

        if not repo_ready:
            console.print(
                f"[yellow]Status for {repo_name}: [bold]not yet run[/bold] "
                f"(Repository clone failed or network unavailable).[/yellow]"
            )
            all_results[repo_name] = {"status": "not yet run", "reason": "Repository unavailable"}
            continue

        # Build index in-memory
        console.print(f"[dim]Indexing repository {target_repo_dir.name}...[/dim]")
        store = MemoryVectorStore()
        embedder = FastHashEmbedding(dim=384)
        chunker = CodeChunker()
        indexer = Indexer(
            vector_store=store,
            embedding_provider=embedder,
            chunker=chunker,
        )
        indexer.index_directory(target_repo_dir)

        pipeline = RetrievalPipeline(vector_store=store, embedding_provider=embedder)
        metrics = evaluate_pipeline_on_dataset(pipeline, queries, eval_modes)

        # Print Rich table
        table = Table(
            title=f"Retrieval Benchmark: {repo_name} ({len(queries)} queries)",
            header_style="bold cyan",
        )
        table.add_column("Retrieval Mode", style="bold")
        table.add_column("Recall@1", justify="right")
        table.add_column("Recall@3", justify="right")
        table.add_column("Recall@5", justify="right")
        table.add_column("Recall@10", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("nDCG@10", justify="right")

        repo_res_dict: dict[str, Any] = {}
        for m_name, res in metrics.items():
            table.add_row(
                m_name,
                f"{res.recall_1:.3f}",
                f"{res.recall_3:.3f}",
                f"{res.recall_5:.3f}",
                f"{res.recall_10:.3f}",
                f"{res.mrr:.3f}",
                f"{res.ndcg_10:.3f}",
            )
            repo_res_dict[m_name] = res.model_dump()

        console.print(table)
        all_results[repo_name] = {
            "status": "completed",
            "metrics": repo_res_dict,
        }

    # Summary note on trade-offs
    console.print(
        "\n[dim yellow]Note on Hybrid vs BM25 trade-offs:[/dim yellow] "
        "[dim]BM25 exhibits near-perfect precision on exact code identifiers, method names, "
        "and error constants. Hybrid fusion (especially RRF) is optimal when queries blend "
        "natural language architectural intent with specific domain terminology.[/dim]"
    )

    if output:
        output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        console.print(f"[green]Saved evaluation results to {output}[/green]")
