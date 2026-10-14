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
- **Code Intelligence Layer (Phase 3)**:
  - Multi-language AST parsing (Python AST + regex support for TypeScript, Go, Rust, Java).
  - SQLite symbol table with definition lookups, class/method hierarchies, and import edges.
  - Static call graph with transitive caller/callee resolution, cycle detection, and Mermaid/DOT exports.
  - Commands: `khaslana where-used <symbol>`, `khaslana graph <symbol>`, and `khaslana explain --deep <file>`.
- **ReAct Agent System (Phase 4)**:
  - Extensible `Tool` base class and `ToolRegistry` with strict Pydantic input/output validation.
  - Core tools: `read_file`, `search_code`, `list_files`, `run_command`, `find_symbol`, `where_used`.
  - ReAct agent loop with max-iteration caps, token budgeting, loop detection, and JSON auto-repair.
  - Mutating shell command security with interactive confirmation prompts.
  - Commands: `khaslana debug <error.log>` and `khaslana analyze project`.
- **Git Integration & Diff-Aware Code Review (Phase 5)**:
  - Git inspector using GitPython for staged/unstaged/branch diffs, blame, and commit history.
  - Diff-aware context enrichment: maps touched diff lines to symbol definitions and resolves callers/callees.
  - Multi-severity code review (`bug`, `security`, `perf`, `style`) with patch generation.
  - Multi-format output for humans and CI pipelines (`--format rich|json|markdown`).
  - Commands: `khaslana review --staged` (or `--branch <name>`), `khaslana explain-diff`, and `khaslana commit-msg`.

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

# 6. Trace symbol definitions and references
khaslana where-used AuthService
khaslana graph authenticate_user --format mermaid

# 7. Agentic debugging and project analysis
khaslana debug error.log
khaslana analyze project

# 8. Git review, diff explanation, and conventional commit generation
khaslana review --staged
khaslana review --branch main --format markdown
khaslana explain-diff --staged
khaslana commit-msg
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
