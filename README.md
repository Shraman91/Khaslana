# Khaslana

A production-grade, local-first AI developer tool in Python.

---

## Features

- **Local-First & Multi-Provider LLM Routing**: Primary local Ollama engine with automatic graceful fallback to Google Gemini and OpenAI.
- **Strict Secret Redaction**: Deterministic regex and Shannon-entropy sanitization prevents leaking credentials or private keys.
- **RAG & Code Retrieval (Phase 1)**:
  - Structure-aware Python AST chunking (classes, methods, module scopes) and token-bounded line window chunking.
  - Persistent vector store powered by ChromaDB under `.khaslana/index/` with in-memory fallback.
  - Pluggable embeddings (`sentence-transformers/all-MiniLM-L6-v2` and fast-hash offline embeddings).
  - Codebase citations with precise `file.py:start-end` line ranges.
- **Hybrid Retrieval & Score Fusion (Phase 2)**:
  - BM25Plus sparse keyword index with code identifier tokenization (camelCase & snake_case sub-token splitting).
  - Score fusion supporting both Weighted normalization (default 0.6 semantic / 0.4 BM25) and Reciprocal Rank Fusion (`rrf`).
  - Contextual priority boosts for focused files (`--file`) and git-modified files.
  - Explainable code search CLI (`khaslana search --explain`) detailing scoring components.

---

## Retrieval Evaluation Benchmark

Evaluation harness results across ground-truth codebase queries on the sample repo:

| Method              | Recall@1 | Recall@3 | Recall@5 |
|---------------------|----------|----------|----------|
| Semantic Only       | 0.50     | 0.88     | 1.00     |
| BM25 Only           | 0.75     | 1.00     | 1.00     |
| **Hybrid (Weighted)** | **0.75** | **1.00** | **1.00** |
| Hybrid (RRF)        | 0.62     | 1.00     | 1.00     |

---

## Installation

```bash
# Using pipx (recommended)
pipx install .

# Or local editable install
pip install -e .
```

---

## Quickstart

```bash
# 1. Check system dependencies and model health
khaslana doctor

# 2. Index your repository into the local vector database
khaslana index .

# 3. Hybrid code search with scoring breakdown
khaslana search "authenticate user" --explain

# 4. Ask questions about your codebase with retrieved context citations
khaslana ask "What does the auth system do?"

# 5. Explain specific files in detail
khaslana explain src/khaslana/retrieval/indexer.py
```

---

## Configuration

Khaslana follows a strict configuration hierarchy:
`CLI Flags > Environment Variables > Project .khaslana.toml > ~/.config/khaslana/config.toml > Defaults`

```bash
# View active configuration
khaslana config show

# Set custom default provider or model
khaslana config set llm.provider ollama
khaslana config set llm.model llama3
```
