# Khaslana

A production-grade, local-first AI developer tool in Python.

---

## Features

- **Local-First & Multi-Provider LLM Routing**: Primary local Ollama engine (`llama3.2`, `qwen2.5`, etc.) with zero cloud data transmission and automatic graceful fallback to Google Gemini and OpenAI.
- **Strict Secret Redaction**: Deterministic regex and Shannon-entropy sanitization prevents leaking credentials, tokens, or private keys before prompts leave the system.
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
- **Sandbox Execution & Self-Correcting Fix Loop (Phase 6)**:
  - Ephemeral Docker execution with strict CPU, memory, and timeout bounds (`network_disabled=True`).
  - Self-correcting loop: applies proposed patches to a sandbox copy, runs tests, and refines iteratively on failure.
  - Zero risk to working tree: host files are modified only after interactive user confirmation (`--yes`).
  - Command: `khaslana fix <file> --test-cmd "<command>" [--max-attempts N]`.
- **Memory System & Style Learning (Phase 7)**:
  - SQLite interaction storage (`.khaslana/memory.db`) with exact cosine semantic recall of past queries and answers.
  - AST `StyleLearner` extracts codebase conventions (naming styles, typing annotations, docstrings, route frameworks).
  - User feedback loop (`khaslana feedback good|bad`) reinforces preferred coding styles.
  - Complete local data sovereignty: `memory list`, `memory search`, `memory forget`, `memory clear`, and `--no-memory`.
- **Background Indexing & Live File Watcher (Phase 8)**:
  - Watchdog-based file observer with configurable debounce windows (`DebouncedEventHandler`).
  - SHA-256 content hashing to eliminate unnecessary re-indexing on touch or metadata changes.
  - Multi-layer incremental synchronization: vector store chunks, AST symbol table, and call graph.
  - Background daemon lifecycle (`khaslana daemon start|stop|status`) with detached process spawning and PID management.
  - Automatic index staleness detection and refresh on `khaslana ask`.
- **Dynamic Plugin System & Built-in Tools (Phase 9)**:
  - Multi-source plugin discovery: Python entry points (`khaslana.plugins`), user directory (`~/.config/khaslana/plugins/`), and workspace plugins (`.khaslana/plugins/`).
  - Full error isolation: malformed or failing plugins cannot crash the CLI; bad plugins are captured with actionable error diagnostics.
  - Built-in tools:
    - `db-query`: Read-only SQLite inspector rejecting mutating SQL commands (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`).
    - `docker`: Safe container inspection (`list`, `logs`, `inspect`) with graceful failure when Docker is offline.
    - `api-tester`: HTTP client with response assertions, latency measurements, and JSON formatting.
  - CLI management: `khaslana plugins list|info|enable|disable` with runtime state persistence (`plugins.json`).
  - Full developer tutorial and guide in [`docs/PLUGINS.md`](docs/PLUGINS.md).

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

## Indexing Benchmarks

Measured on the test codebase (6 files, 14 semantic chunks):

| Index Operation | Latency | Speedup | Notes |
|---|---|---|---|
| **Full Repository Index** | `21.25 ms` | `1.0x` | Full AST parse, chunking, and vector generation |
| **Incremental No-op Scan** | `9.60 ms` | `2.2x` | Rapid SHA-256 hash check over all tracked files |
| **Single Modified File Update** | `0.74 ms` | **28.7x** | Delta update (re-chunk, re-embed, update Chroma & SQLite) |

---

## Installation & Setup

### Option 1: Local Virtual Environment (Fastest)
```powershell
# Clone the repository
git clone https://github.com/Shraman91/Khaslana.git
cd Khaslana

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Option 2: Global PowerShell Alias
To make `khaslana` accessible from any folder without re-downloading ML packages:
```powershell
Add-Content -Path $PROFILE -Value 'function khaslana { & "c:\path\to\khaslana\.venv\Scripts\khaslana.exe" @args }' -Force
. $PROFILE
```

### Option 3: Global Install via `pipx`
```powershell
pipx install "c:\path\to\khaslana"
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
khaslana explain --deep src/khaslana/core/llm.py

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

# 9. Sandbox execution & self-correcting fix loop
khaslana fix src/utils.py --test-cmd "pytest tests/test_utils.py"
khaslana fix src/api.py --max-attempts 3 --yes

# 10. Memory system & coding style
khaslana ask "how do I write API endpoints?"              # Injects style profile & past answers
khaslana ask "how do I write API endpoints?" --no-memory   # Bypass memory injection & logging
khaslana feedback good                                     # Record positive feedback on last response
khaslana feedback bad --id <query-id>                      # Rate specific past interaction
khaslana memory list                                       # Inspect past queries & recorded fixes
khaslana memory search "database connection"               # Semantic recall search over past interactions
khaslana memory profile                                    # View codebase AST style rules & patterns
khaslana memory forget <id>                                # Delete specific interaction from memory
khaslana memory clear --yes                                # Erase all local memory data
khaslana memory export --out memory_backup.json            # Export interaction history to JSON

# 11. Background indexing & live file watching
khaslana watch                                             # Live foreground watcher with debounced re-indexing
khaslana watch --debounce 1.0                              # Custom debounce window (seconds)
khaslana daemon start                                      # Start detached background indexing daemon
khaslana daemon status                                     # Check background daemon PID and uptime
khaslana daemon stop                                       # Gracefully terminate background indexing daemon

# 12. Plugin management & extension inspection
khaslana plugins list                                      # List discovered plugins, versions, and tools
khaslana plugins info db-query                             # Inspect parameters & schemas for plugin tools
khaslana plugins disable docker                            # Disable a plugin without removing files
khaslana plugins enable docker                             # Re-enable a plugin
```

---

## Testing

```powershell
# Run the complete test suite (138 tests)
pytest

# Run the end-to-end smoke test script across all subsystems
.\scripts\smoke_test.ps1

# (On Linux / macOS / Bash)
./scripts/demo.sh

# Run code style and type safety checks
ruff check src tests
mypy src tests
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
khaslana config set llm.model llama3.2
```

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Full component diagram and subsystem data flows.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): Architectural Decision Records (ADRs 001–011).
- [`docs/PLUGINS.md`](docs/PLUGINS.md): "Write your own plugin in 5 minutes" tutorial, interfaces, and templates.
