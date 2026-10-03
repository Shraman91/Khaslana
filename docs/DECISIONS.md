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
