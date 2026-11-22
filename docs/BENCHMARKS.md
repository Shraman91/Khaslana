# Retrieval Evaluation Benchmarks

This document describes the information retrieval evaluation methodology, benchmark harness, and metrics for Khaslana.

## Objective

Khaslana indexes source code into chunks containing syntactic boundaries (classes, functions, methods, and interface definitions) with symbol metadata.
To rigorously measure retrieval effectiveness, Khaslana implements the `khaslana eval` harness evaluating 4 retrieval paradigms:

1. **Semantic Vector Search**: Dense vector search via cosine distance.
2. **BM25**: Lexical term-frequency / inverse-document-frequency ranking.
3. **Hybrid-Weighted**: Convex linear combination of min-max normalized semantic similarity and BM25 scores ($0.6 \cdot \text{Semantic} + 0.4 \cdot \text{BM25}$).
4. **Hybrid-RRF**: Reciprocal Rank Fusion ($\frac{1}{60 + \text{rank}}$), combining dense and sparse results without requiring raw score calibration.

---

## Benchmark Datasets

The evaluation suite defines 3 target repositories with pinned commits and 40+ curated developer queries each (located in `benchmarks/`):

| Repository | Language | Pinned Commit | Queries | Benchmark Spec |
|------------|----------|---------------|---------|----------------|
| **Click** (`pallets/click`) | Python | `e396652` | 42 | [`benchmarks/python_repo.json`](../benchmarks/python_repo.json) |
| **Got** (`sindresorhus/got`) | TypeScript | `479c884` | 42 | [`benchmarks/typescript_repo.json`](../benchmarks/typescript_repo.json) |
| **Gin** (`gin-gonic/gin`) | Go | `0067645` | 42 | [`benchmarks/go_repo.json`](../benchmarks/go_repo.json) |

> [!IMPORTANT]
> **Ground Truths Require Ongoing Human Review**
> Developer queries span high-level architectural lookups, specific middleware handlers, decorator definitions, and error handling. Because complex codebases feature cross-cutting concerns, ground-truth labels require ongoing human review and validation.

---

## Evaluation Metrics

For each query, the retrieval system returns the top $K=10$ candidate chunks. The harness computes:

- **Recall@K ($K \in \{1, 3, 5, 10\}$)**: Proportion of queries where at least one ground-truth file or symbol is retrieved within the top $K$ items.
- **MRR (Mean Reciprocal Rank)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
  where $\text{rank}_q$ is the 1-based rank of the first relevant retrieved chunk ($0$ if no hit within top 10).
- **nDCG@10 (Normalized Discounted Cumulative Gain)**:
  $$\text{DCG@10} = \sum_{i=1}^{10} \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{nDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$
  where $\text{rel}_i \in \{0, 1\}$ indicates binary relevance.

---

## Benchmark Results

> [!NOTE]
> In hermetic or network-isolated environments where remote repository cloning is disabled, `khaslana eval` marks remote benchmarks as **not yet run** rather than fabricating numbers.

| Dataset | Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 | Status |
|---|---|---|---|---|---|---|---|---|
| **Click (Python)** | BM25 | — | — | — | — | — | — | *not yet run* |
| Click (Python) | Semantic | — | — | — | — | — | — | *not yet run* |
| Click (Python) | Hybrid-Weighted | — | — | — | — | — | — | *not yet run* |
| Click (Python) | Hybrid-RRF | — | — | — | — | — | — | *not yet run* |
| **Got (TypeScript)** | BM25 | — | — | — | — | — | — | *not yet run* |
| Got (TypeScript) | Semantic | — | — | — | — | — | — | *not yet run* |
| Got (TypeScript) | Hybrid-Weighted | — | — | — | — | — | — | *not yet run* |
| Got (TypeScript) | Hybrid-RRF | — | — | — | — | — | — | *not yet run* |
| **Gin (Go)** | BM25 | — | — | — | — | — | — | *not yet run* |
| Gin (Go) | Semantic | — | — | — | — | — | — | *not yet run* |
| Gin (Go) | Hybrid-Weighted | — | — | — | — | — | — | *not yet run* |
| Gin (Go) | Hybrid-RRF | — | — | — | — | — | — | *not yet run* |

---

## Honest Analysis: When Does Hybrid NOT Beat BM25?

A common misconception in code retrieval is that hybrid retrieval strictly dominates lexical search across all queries. In real software development:

1. **Exact Symbol and Constant Lookups**:
   - Queries like `SIGTERM`, `BadRequestError`, `404`, or `RedirectTrailingSlash` are unambiguous tokens.
   - **BM25** scores these with near-perfect Recall@1 and MRR because exact token matches receive high inverse-document-frequency boosts.
   - **Semantic vectors** project tokens into dense conceptual space, which can introduce semantic drift (ranking related exception types or HTTP handlers above the exact identifier). In such cases, hybrid weighting can dilute BM25's exact top-1 ranking.
2. **Short Identifier Queries**:
   - For single-token queries (e.g. `NewServer`, `CliRunner`), BM25's exact inverted index is faster and sharper than embedding-based similarity.
3. **Where Hybrid & RRF Excel**:
   - Hybrid fusion (specifically **RRF**) significantly outperforms BM25 when queries express *intent* or *synonyms* that don't match the exact identifiers in the code (e.g. "intercept outgoing requests before sending" vs `beforeRequestHook`).

---

## How to Run the Benchmark Locally

```bash
# Evaluate benchmark datasets
khaslana eval

# Evaluate a specific dataset
khaslana eval --dataset benchmarks/python_repo.json

# Compare specific retrieval modes
khaslana eval --modes semantic,bm25,hybrid-rrf --output eval_results.json
```
