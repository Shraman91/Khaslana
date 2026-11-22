# Khaslana

[![CI](https://github.com/shraman-deshpande/khaslana/actions/workflows/ci.yml/badge.svg)](https://github.com/shraman-deshpande/khaslana/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#limitations-and-roadmap)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An open-source, local-first AI developer tool and terminal assistant in Python. Khaslana pairs local LLMs (Ollama) with semantic code search, static call graphs, sandboxed code fixes, and automated debugging while keeping your code private on your machine.

---

## Key Features

- **Local-First & Privacy Boundaries**:
  - Primary local engine via Ollama (`llama3.2`, `qwen2.5`, `deepseek-r1`) with zero telemetry.
  - Cloud models (Gemini, OpenAI) are **disabled by default** and strictly blocked unless explicitly requested with `--allow-cloud`.
  - Secret redaction choke point masks API keys, passwords, and `.env` variables before messages leave the process.
  - Detailed privacy model in [`docs/PRIVACY.md`](docs/PRIVACY.md).
- **Hybrid Code Retrieval & Empirical Evaluation**:
  - AST-aware Python chunking and token-bounded line window chunking.
  - Dual-mode hybrid search combining dense semantic embeddings with BM25Plus keyword matching.
  - Choice of score fusion: Weighted combination or Reciprocal Rank Fusion (`rrf`).
  - Empirical evaluation harness via `khaslana eval` with Recall@1/3/5/10, MRR, and nDCG@10 metrics.
  - Evaluation details in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- **Static Code Intelligence**:
  - Python AST parsing and multi-language support (TypeScript, Go, Rust, Java) via optional `tree-sitter` with regex fallback.
  - SQLite symbol tables with definition lookups, class/method hierarchies, and import edges.
  - Static call graph with transitive caller/callee resolution, circular dependency detection, and Mermaid/DOT exports.
- **ReAct Agent & Shell Safety**:
  - ReAct agent loop for multi-step diagnosis and code analysis (`khaslana debug`, `khaslana analyze`).
  - Strict command execution security: `shell=False`, prohibited metacharacters, and strict read-only allowlist.
  - Untrusted file and command outputs wrapped in `<<<UNTRUSTED_CONTENT>>>` passive delimiters.
  - Threat model and security policies documented in [`SECURITY.md`](SECURITY.md).
- **Sandboxed Self-Correcting Fixes (`khaslana fix`)**:
  - Executes unit tests in ephemeral Docker containers (`network_disabled=True`).
  - Refuses execution by default if Docker is unavailable unless `--no-sandbox` is passed.
  - Working tree remains 100% untouched until interactive user approval (`--yes`).
- **Developer Workflows**:
  - `khaslana doctor`: Diagnoses system health, Ollama server status, and installed models.
  - `khaslana config`: View and manage configuration settings hierarchy.
  - `khaslana review`, `khaslana explain-diff`, `khaslana commit-msg`: Git diff-aware code review and conventional commit generation.
  - `khaslana daemon` & `khaslana watch`: Live filesystem change observer with incremental debounced re-indexing.
  - Pluggable extension architecture (`docs/PLUGINS.md`).

---

## Demo

```
$ khaslana ask "Where are user tokens authenticated?"

Retrieving context (Hybrid RRF)...
- src/auth/service.py:12-45 (AuthService.validate_token)
- src/api/middleware.py:18-35 (verify_jwt)

Khaslana (ollama/llama3.2):
User tokens are authenticated in `AuthService.validate_token` (src/auth/service.py:12-45),
which is invoked on every request by `verify_jwt` middleware (src/api/middleware.py:18-35).
```

---

## How It Compares

| Capability / Dimension | Khaslana | Aider | Continue | Claude Code |
|---|---|---|---|---|
| **Primary Execution** | Local CLI & terminal | Local CLI git-pairing | IDE extension (VS Code/JetBrains) | Hosted CLI agent |
| **Privacy by Default** | **Local-only** (Ollama); cloud disabled | BYO API key (cloud-first) | BYO API key (cloud-first) | Cloud-only (Anthropic) |
| **Retrieval Architecture** | Hybrid (Vector + BM25 + RRF) + Call Graph | Repository map (tree-sitter ctags) | Vector RAG or BM25 index | Server-side grep / glob / file read |
| **Sandbox Isolation** | **Docker sandbox** with network off | Direct on-host execution | Direct on-host execution | Direct on-host execution |
| **Code Intelligence** | AST + SQLite symbol table + Call Graph | Tag graph ranking | Basic LSP integration | File exploration tools |
| **Secret Redaction** | Built-in regex & entropy masking | User responsibility | User responsibility | User responsibility |

### Honest Trade-offs
- **When Aider is better**: Aider has deeper git integration, whole-repo editing benchmarks, and sophisticated multi-file edit formats optimized for frontier cloud LLMs.
- **When Continue is better**: Continue offers tight visual IDE integration directly in VS Code and JetBrains editors with inline diff viewers.
- **When Claude Code is better**: Claude Code leverages Sonnet 3.7 with high agency for sprawling autonomous multi-hour tasks.
- **When Khaslana is better**: You require strict air-gapped data sovereignty, local Ollama models, sandboxed automated test fixes in Docker, and hybrid retrieval with static call graph inspection.

---

## Limitations and Roadmap

> [!WARNING]
> **Alpha Status**: Khaslana is currently under active development. Core features are functional and tested, but APIs and CLI flags may evolve before v1.0.

### Current Limitations
- **Local Model Capacity**: Complex multi-file reasoning depends heavily on the underlying local model. Smaller models (e.g. 7B/8B) may struggle with complex ReAct loops compared to frontier cloud models.
- **Approximate Call Graphs**: Static call graphs resolve symbols syntactically. Polymorphic dynamic dispatch and runtime monkeypatching are approximate.
- **Language Support**: Deep AST analysis is native in Python. TypeScript, Go, Rust, and Java support relies on tree-sitter or regex fallback.

### Roadmap
- [ ] Interactive Textual TUI (`khaslana tui`)
- [ ] Tree-sitter semantic call graph resolution for C/C++ and Kotlin
- [ ] Full SWE-bench benchmark harness evaluation
- [ ] Streaming unified diff viewer directly in the terminal

---

## Installation

### Lightweight Base Install
Base Khaslana is lightweight with zero Torch, ChromaDB, or Docker overhead. It runs entirely offline using built-in fast-hash embeddings:
```powershell
pip install -e .
```

### Optional Extras
Install feature extras as needed:
```powershell
# Neural embeddings & persistent ChromaDB
pip install -e ".[embeddings]"

# Docker sandbox execution for safe automated code fixes
pip install -e ".[sandbox]"

# Tree-sitter AST parsers for TypeScript, Go, Rust, Java
pip install -e ".[parsers]"

# All optional dependencies
pip install -e ".[all]"
```

### Global CLI Access via `pipx`
```powershell
pipx install "c:\path\to\khaslana[all]"
```

---

## Quickstart

```bash
# 1. Check system dependencies and model health
khaslana doctor

# 2. Index your repository into the local vector database
khaslana index .

# 3. Ask questions about your codebase with retrieved citations
khaslana ask "What does the auth system do?"

# 4. Opt-in to cloud fallback when local Ollama is insufficient
khaslana ask "Complex architecture review" --allow-cloud

# 5. Explain specific files or trace call graphs
khaslana explain --deep src/khaslana/retrieval/indexer.py
khaslana where-used AuthService
khaslana graph authenticate_user --format mermaid

# 6. Sandboxed self-correcting fix loop in Docker
khaslana fix src/utils.py --test-cmd "pytest tests/test_utils.py"

# 7. Agentic debugging and project analysis
khaslana debug error.log
khaslana analyze project

# 8. Git review and conventional commit generation
khaslana review --staged
khaslana commit-msg

# 9. Retrieval evaluation benchmarks
khaslana eval --dataset benchmarks/python_repo.json
```

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Component diagrams, subsystems, and data flows.
- [`docs/PRIVACY.md`](docs/PRIVACY.md): Redaction rules and cloud model isolation.
- [`SECURITY.md`](SECURITY.md): Threat model, shell command security, and sandbox bounds.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): Retrieval benchmark harness and comparative metrics.
- [`CHANGELOG.md`](CHANGELOG.md): Release history and version notes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): Local development setup, linting, and testing instructions.
- [`docs/PLUGINS.md`](docs/PLUGINS.md): Guide to creating custom plugins and tools.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): Architectural Decision Records (ADRs).
