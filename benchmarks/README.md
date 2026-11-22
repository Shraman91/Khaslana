# Retrieval Evaluation Benchmark Suite

This directory contains evaluation specifications and queries for assessing Khaslana's retrieval pipeline across multiple programming languages and codebases.

## Datasets

The suite targets 3 widely-used open-source repositories pinned to deterministic commits:

1. **Python**: [`pallets/click`](python_repo.json) (commit `e39665261eb8c9e6bb0702c2e6f499690623dd3f`) - 42 queries.
2. **TypeScript**: [`sindresorhus/got`](typescript_repo.json) (commit `479c8843997dbcb9a36ff9aa50f1469e7be94018`) - 42 queries.
3. **Go**: [`gin-gonic/gin`](go_repo.json) (commit `006764585eb2e3a0979cb391e0a9ce5fd8cbbbe6`) - 42 queries.

## Ground Truth Verification

> [!IMPORTANT]
> **Ground Truths Require Human Review**
> The queries in these benchmark specifications represent realistic developer lookups (architectural concepts, decorators, middleware handlers, type validations, and error types).
> Because real codebases contain overlapping functionality across files, ground truth associations require ongoing human review and domain verification.

## Evaluated Retrieval Modes

`khaslana eval` benchmarks 4 retrieval strategies side-by-side:

- **Semantic**: Dense vector retrieval using local embeddings (`all-MiniLM-L6-v2` or `FastHashEmbedding`).
- **BM25**: Lexical sparse token matching over source code chunks.
- **Hybrid-Weighted**: Convex linear combination of min-max normalized semantic similarity and BM25 scores.
- **Hybrid-RRF**: Reciprocal Rank Fusion (`1 / (60 + rank)`), eliminating score calibration imbalances between dense and sparse ranks.

## Standard Information Retrieval Metrics

- **Recall@K (K=1, 3, 5, 10)**: Proportion of queries where at least one ground-truth file or symbol appears in the top $K$ results.
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks ($\frac{1}{\text{rank}}$) of the first retrieved relevant document.
- **nDCG@10 (Normalized Discounted Cumulative Gain)**: Discounted ranking measure penalizing relevant results placed at lower positions:
  $$\text{DCG@10} = \sum_{i=1}^{10} \frac{\text{rel}_i}{\log_2(i + 1)}$$

## Running the Benchmark

```bash
# Evaluate all benchmark datasets
khaslana eval

# Evaluate a specific language dataset
khaslana eval --dataset benchmarks/python_repo.json

# Compare specific retrieval modes
khaslana eval --modes bm25,hybrid-rrf
```

Cloned repositories are cached in `~/.khaslana/cache/eval_repos/` (or `--cache-dir`). If network access is unavailable, the evaluation runner skips missing remote repositories with an explicit status rather than inventing scores.
