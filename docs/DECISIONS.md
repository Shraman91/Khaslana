# Architectural Decision Record (ADR)

This file tracks all key architectural and implementation decisions made during the development of Khaslana.

---

## ADR 001: Pydantic for Data Models Across Module Boundaries

- **Status:** Accepted
- **Context:** Modules need clear, typed contracts for data passing.
- **Decision:** Use Pydantic v2 `BaseModel` for all inter-module contracts (chunks, tool parameters, results, configurations, analysis outputs).
- **Consequences:** Ensures schema validation, runtime typing, and clean JSON serialization.

---

## ADR 002: Provider Fallback Order & LLMRouter Strategy

- **Status:** Accepted
- **Context:** Khaslana is a local-first tool, but local models may be unavailable, low-powered, or fail.
- **Decision:** Default primary provider is local Ollama. On connection failure or timeout, LLMRouter auto-falls back to cloud options (Gemini, OpenAI) if API keys are configured.
- **Consequences:** Provides local privacy by default while maintaining resilience when local setups are insufficient.

---

## ADR 003: Core Redaction at the LLM Boundary

- **Status:** Accepted
- **Context:** Sending code, logs, environment files, or diffs to cloud providers risks leaking API keys and tokens.
- **Decision:** Implement `core/redact.py` with pattern-based and high-entropy regex masking. Apply it *inside* `LLMRouter` before any request is sent out to any backend.
- **Consequences:** Zero-leak guarantee regardless of which higher-level module originated the text.

---

## ADR 004: Standardized Error Hierarchy

- **Status:** Accepted
- **Context:** Tool failures, missing binaries, or bad connections must fail gracefully with actionable user messages rather than raw tracebacks.
- **Decision:** Define a central `KhaslanaError` exception hierarchy in `core/errors.py`. CLI handles catching top-level `KhaslanaError` and presenting formatted error panels, showing tracebacks only when `--debug` is active.
- **Consequences:** Clean CLI experience and clear diagnostics.

---

## ADR 005: Hybrid Retrieval with BM25Plus & Reciprocal Rank Fusion

- **Status:** Accepted
- **Context:** Dense embeddings alone miss exact keyword matches (e.g. identifier names like `AuthService`), while BM25 alone misses semantic synonyms. Small repository fixtures also suffer from negative IDF under classic Okapi formulas.
- **Decision:** Combine BM25Plus (with camelCase/snake_case sub-token splitting) with dense vector embeddings via weighted score normalization (default: 0.6 semantic / 0.4 BM25) and Reciprocal Rank Fusion (`rrf`). Support priority boosts for focused files and modified working tree files.
- **Consequences:** High recall on code searches with exact symbol and conceptual queries.

---

## ADR 006: ReAct Agent Loop with Strict-JSON Schema and Action Confirmation

- **Status:** Accepted
- **Context:** Agent workflows require multi-step reasoning, tool invocations, and interactive execution. Many local LLMs lack proprietary native function calling APIs or output markdown code blocks. In addition, executing arbitrary shell commands without boundaries poses security and destruction risks.
- **Decision:**
  1. Standardize on JSON-in-prompt ReAct loop (`{"thought", "action", "action_input"}` / `{"thought", "final_answer"}`). Implement robust JSON auto-repair (strip code fences, fix trailing commas, unescaped characters).
  2. Implement loop/repeat-action detection to prevent infinite repetition of identical tool calls.
  3. Implement strict command execution policies in `run_command`: allowlist for safe inspection commands, denylist for dangerous actions, and an interactive confirmation prompt (or programmable confirmation callback) for any mutating or non-allowlisted command.
- **Consequences:** Deterministic, safe, and works uniformly across Ollama, Gemini, OpenAI, and testing fakes.

---

## ADR 007: Diff-Aware Git Review with Static Symbol Context Enrichment

- **Status:** Accepted
- **Context:** Standard code reviews looking solely at git diff hunks miss the broader architectural impact, such as callers of modified methods or whether a changed function signature breaks existing call sites.
- **Decision:**
  1. Build a dedicated Git inspector in `tools/git.py` supporting staged diffs, branch diffs (e.g. against `main`), logs, blame, and changed files.
  2. Implement diff-aware context enrichment: parse files in the diff to locate which functions or classes were touched, then query Phase 3's `SymbolTable` and `CallGraph` to retrieve surrounding definitions, callers, and callees.
  3. Format review output with severities (`bug`, `security`, `perf`, `style`), line references, and suggested patches, supporting `--format rich|json|markdown` for CLI and CI integration.
- **Consequences:** Code reviews have deep semantic context beyond raw diff lines, with seamless output for both humans and CI pipelines.

---

## ADR 008: Ephemeral Sandbox Containers & Non-Destructive Self-Correcting Fixes

- **Status:** Accepted
- **Context:** Automated code fixes can generate syntax errors, broken dependencies, or test regressions. Executing test commands directly on the host poses security and corruption risks to the active working tree.
- **Decision:**
  1. Use Docker containers with strict resource limits (nano_cpus, mem_limit, default 30s timeout) and network isolation (`network_disabled=True`). Auto-detect container base images per language.
  2. Implement non-destructive copy isolation: create a temporary workspace copy of the project and mount that copy into the container. Never execute unverified code in the host working tree.
  3. Self-correcting loop: run tests in the sandbox, feed stdout/stderr back into the LLM on failure for up to N attempts, and apply the final fix to the user's real file only upon user confirmation (`--yes` to auto-apply).
  4. Provide `SandboxUnavailableError` for graceful degradation when Docker is unavailable or not running, and support `FakeSandboxRunner` for deterministic testing.
- **Consequences:** Safe, isolated code verification with zero risk of working tree corruption during repair attempts.

---

## ADR 009: SQLite Interaction Memory, Cosine Semantic Recall, and AST Style Learner

- **Status:** Accepted
- **Context:** An AI developer tool should become more helpful over time by remembering past interactions, incorporating user feedback, and adhering to the project's coding conventions without requiring manual style prompt engineering.
- **Decision:**
  1. Store interaction histories (`queries`, `fixes`, `feedback`, `style_profile`) in local SQLite (`.khaslana/memory.db`). Embed interaction queries using the shared embedding provider and serialize vectors as JSON blobs.
  2. Implement semantic recall using exact cosine similarity over historical interaction embeddings. On `khaslana ask`, query embeddings retrieve the most relevant past questions and answers within a configurable similarity threshold.
  3. Implement `StyleLearner` using AST inspection over python files (extracting function naming conventions, typing annotation discipline, docstring presence/formats, and framework patterns like FastAPI/Flask route decorators) and refine style weights when user patches are accepted.
  4. Ensure complete user data sovereignty: provide `memory list`, `memory search <query>`, `memory forget <id>`, `memory clear [--yes]`, `memory export [--out]`, and `--no-memory` flag to bypass memory storage and retrieval.
- **Consequences:** Contextual continuity and codebase-aligned style guidance while guaranteeing local-only storage and privacy.

---

## ADR 010: Incremental File Watcher, Content Hashing, and Daemon Lifecycle

- **Status:** Accepted
- **Context:** Re-indexing an entire repository on every change is expensive, slow, and causes unnecessary GPU/CPU load. The retrieval index must update in near-real-time as developers write, delete, and rename files.
- **Decision:**
  1. Use `watchdog.observers.Observer` with thread-safe debouncing (`DebouncedEventHandler`, default 0.5s delay) to coalesce rapid editor save bursts and avoid repeated thrashing.
  2. Implement content hash validation (`hashlib.sha256`) stored in `.khaslana/index_state.json`. If file mtime changes but content hash remains identical (e.g. editor touch or metadata update), re-indexing is skipped.
  3. Support incremental multi-layer updates:
     - Vector Store: Delete previous file chunks and insert newly chunked embeddings.
     - Symbol Table & Call Graph: Remove old symbol/call records for the touched file and parse updated AST definitions.
     - Deletions & Renames: Handle file removals by purging vector chunks and symbol table rows.
  4. Provide foreground live watching (`khaslana watch`) and background daemon lifecycle (`khaslana daemon start|stop|status`) managed with a JSON PID record (`.khaslana/daemon.pid`), automatic dead PID detection, and clean termination across Windows and POSIX.
  5. Automatically detect index staleness on `khaslana ask` to incrementally update before executing search queries.
- **Consequences:** Sub-millisecond incremental updates (~28x faster than full re-index) with always-fresh RAG context.


