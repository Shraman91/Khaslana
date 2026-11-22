# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Optional Dependency Extras**: Heavy packages isolated behind optional extras:
  - `embeddings`: `sentence-transformers`, `chromadb`
  - `sandbox`: `docker`
  - `parsers`: `tree-sitter>=0.20.0`
  - `all`: installs all optional feature groups.
- **Privacy & Redaction Enforcement**:
  - Cloud fallback disabled by default (`allow_cloud_fallback=false`).
  - `--allow-cloud` CLI flag and `KHASLANA_ALLOW_CLOUD` environment variable for explicit cloud opt-in.
  - Prominent stderr warning emitted whenever cloud models are used.
  - Single choke point in `LLMClient.chat()` and `LLMClient.stream()` ensuring secret sanitization precedes all outbound traffic.
  - False-positive resistance preserving 40-character git commit hashes, UUIDs, and SHA-256 hashes.
  - `docs/PRIVACY.md` documentation guide.
- **Strict Shell Security & Sandbox Guards**:
  - Replaced shell execution with `shell=False` and `shlex.split()`.
  - Disallowed shell metacharacters, pipes, redirects, subshells, backticks, and chain operators.
  - Blocked shell wrapper binaries (`sh`, `bash`, `cmd`, `powershell`, `eval`, `sudo`, `env`).
  - Prohibited unsafe git configuration flags (`-c`, `--exec-path`, `--config-env`).
  - Strict read-only allowlist for agent command execution with interactive confirmation for mutating commands.
  - Wrapped untrusted external command and file outputs in `<<<UNTRUSTED_CONTENT>>>` delimiters.
  - `khaslana fix` now requires Docker by default, refusing execution unless `--no-sandbox` is explicitly provided.
  - `LocalProcessRunner` fallback executing tests against temporary workspace copies when running with `--no-sandbox`.
  - `SECURITY.md` security policy and threat model documentation.
- **Multi-Language Parser Hardening**:
  - Optional `tree-sitter` integration with graceful single-warning regex fallback.
  - Robust regex parsing for TypeScript, Go, Rust, and Java handling nested generics, multi-line signatures, Go receiver methods, Rust `impl` blocks and `macro_rules!`, and TypeScript arrow functions.
  - Documented approximations for static call graph analysis under dynamic dispatch and polymorphic calls.
  - Tricky syntax fixtures and comprehensive unit tests for all four languages.
- **Retrieval Evaluation Benchmark**:
  - `khaslana eval` CLI command for empirical information retrieval benchmarking.
  - Standard IR metrics: Recall@1, Recall@3, Recall@5, Recall@10, MRR, and nDCG@10.
  - Side-by-side evaluation across Semantic, BM25, Hybrid-Weighted, and Hybrid-RRF strategies.
  - Deterministic benchmark specifications for Python (`Click`), TypeScript (`Got`), and Go (`Gin`) in `benchmarks/`.
  - `docs/BENCHMARKS.md` retrieval benchmark methodology and trade-off documentation.
- **CI & Quality Assurance**:
  - Multi-platform GitHub Actions workflow matrix (`ubuntu-latest`, `windows-latest` on Python 3.11 and 3.12).
  - Custom pytest markers for `docker`, `ollama`, and `network`.

### Changed
- Migrated configuration parser from third-party `toml` to Python standard library `tomllib`.
- Explicit author metadata ("Shraman Deshpande") and project URLs in `pyproject.toml`.
- Re-labeled project from "production-grade" to **Alpha** in documentation.

## [0.1.0-alpha] - 2026-10-20

### Added
- Initial alpha release of Khaslana local-first AI developer CLI.
- Hybrid vector & BM25 retrieval pipeline with ChromaDB and in-memory fallback.
- Static code intelligence with AST parsing, SQLite symbol table, and call graph queries.
- ReAct agent loop for automated debugging and multi-step investigation.
- Git workflow integrations (`review`, `explain-diff`, `commit-msg`).
- Self-correcting fix loop in Docker sandbox.
- Pluggable tool and plugin system.
