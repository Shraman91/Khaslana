# Khaslana Architecture

## Overview

Khaslana is a production-grade, local-first AI developer tool. It is designed to be modular, testable, and extensible — closer to a platform than a script.

---

## Component Diagram

```mermaid
graph TD
    CLI["CLI Layer\n(typer + rich)\ncli/main.py"]
    TUI["TUI Layer\n(textual)\ntui/app.py"]

    CLI --> Core
    TUI --> Core

    subgraph Core["Core Services"]
        LLM["LLMRouter\ncore/llm.py"]
        CFG["Config\ncore/config.py"]
        LOG["Logging\ncore/logging.py"]
        ERR["Errors\ncore/errors.py"]
        RED["Redact\ncore/redact.py"]
        PRO["Prompts\ncore/prompts.py"]
    end

    LLM -->|"routes via"| OLL["OllamaClient"]
    LLM -->|"falls back to"| GEM["GeminiClient"]
    LLM -->|"falls back to"| OAI["OpenAIClient"]
    LLM --> RED

    subgraph Agents["Agent System"]
        LOOP["AgentLoop\nagents/loop.py"]
        PLAN["Planner\nagents/planner.py"]
        TR["ToolRouter\nagents/tool_router.py"]
    end

    LOOP --> PLAN
    LOOP --> TR
    TR --> Tools

    subgraph Tools["Tool Layer"]
        TB["Tool Base Class\ntools/base.py"]
        RF["read_file"]
        SC["search_code"]
        LF["list_files"]
        RC["run_command"]
        GT["git tools"]
    end

    subgraph Retrieval["RAG / Retrieval"]
        IDX["Indexer\nretrieval/indexer.py"]
        CHK["Chunker\nretrieval/chunker.py"]
        EMB["Embeddings\nretrieval/embeddings.py"]
        STO["VectorStore (Chroma)\nretrieval/store.py"]
        BM25["BM25 Index\nretrieval/bm25.py"]
        RNK["Hybrid Ranker\nretrieval/ranker.py"]
        WAT["File Watcher\nretrieval/watcher.py"]
    end

    subgraph Intelligence["Code Intelligence"]
        PAR["Parser\nintelligence/parser.py"]
        SYM["Symbol Table\nintelligence/symbols.py"]
        CG["Call Graph\nintelligence/callgraph.py"]
    end

    subgraph Memory["Memory System"]
        MST["SQLite Store\nmemory/store.py"]
        STY["Style Profile\nmemory/style.py"]
    end

    subgraph Sandbox["Sandbox Execution"]
        DR["Docker Runner\nsandbox/docker_runner.py"]
    end

    subgraph Merge["Merge Resolution"]
        MP["Parser\nmerge/parser.py"]
        MC["Context Builder\nmerge/context.py"]
        MR["Resolver\nmerge/resolver.py"]
        MV["Validator\nmerge/validate.py"]
    end

    subgraph Deploy["Deployment Doctor"]
        DD["Detector\ndeploy/detect.py"]
        DS["Signatures\ndeploy/signatures.py"]
        DC["Collectors\ndeploy/collectors.py"]
        DG["Diagnose\ndeploy/diagnose.py"]
        DF["Fixers\ndeploy/fixers.py"]
    end

    subgraph Plugins["Plugin System"]
        PL["Loader\nplugins/loader.py"]
        PE["Example Plugins"]
    end

    Tools --> Retrieval
    Tools --> Intelligence
    Agents --> Memory
    Agents --> Core

    style Core fill:#1a1a2e,color:#e0e0e0
    style Agents fill:#16213e,color:#e0e0e0
    style Tools fill:#0f3460,color:#e0e0e0
    style Retrieval fill:#0d2137,color:#e0e0e0
```

---

## Data Flow: `khaslana ask "<query>"`

```
User query
    │
    ▼
CLI (typer) → validate args, load Config
    │
    ▼
Redact (sanitize query)
    │
    ▼
Retrieval pipeline
  ├─ Embed query → vector search (Chroma)
  ├─ BM25 keyword search
  └─ Hybrid rank → Top-K chunks (with file:line citations)
    │
    ▼
Prompt assembly (chunks + system prompt + user query)
    │
    ▼
LLMRouter → OllamaClient (primary)
          → GeminiClient (fallback)
          → OpenAIClient (fallback)
    │
    ▼
Streamed output → rich console
```

---

## Configuration Precedence

```
CLI flags > Environment variables > project .khaslana.toml > ~/.config/khaslana/config.toml > defaults
```

---

## Key Design Decisions

See [`DECISIONS.md`](DECISIONS.md) for rationale behind architectural choices.

---

## Phase Status

| Phase | Feature                   | Status  |
|-------|---------------------------|---------|
| 0     | Foundation + LLM routing  | ✅ Done |
| 1     | RAG System                | ✅ Done |
| 2     | Hybrid Retrieval          | 🔲 TODO |
| 3     | Code Intelligence         | 🔲 TODO |
| 4     | Agent System              | 🔲 TODO |
| 5     | Git Integration           | 🔲 TODO |
| 6     | Sandbox Execution         | 🔲 TODO |
| 7     | Memory System             | 🔲 TODO |
| 8     | Background Indexing       | 🔲 TODO |
| 9     | Plugin System             | 🔲 TODO |
| 10    | Merge Conflict Resolution | 🔲 TODO |
| 11    | Deployment Doctor         | 🔲 TODO |
| 12    | TUI                       | 🔲 TODO |
