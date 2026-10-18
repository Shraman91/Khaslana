"""Configuration management for Khaslana.

Precedence order:
CLI flags > Environment variables > Project `.khaslana.toml`
> User `~/.config/khaslana/config.toml` > Defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import toml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider and connection parameters."""

    provider: str = Field(
        default="ollama",
        description="Primary LLM provider (ollama, gemini, openai, fake)",
    )
    model: str = Field(default="llama3", description="Default model name")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI Base URL")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")
    temperature: float = Field(default=0.2, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Max response tokens")
    timeout_seconds: float = Field(default=60.0, description="HTTP request timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum retries for failed requests")


class RetrievalConfig(BaseModel):
    """Retrieval and RAG parameters."""

    fusion: Literal["weighted", "rrf"] = Field(default="weighted", description="Fusion strategy")
    semantic_weight: float = Field(default=0.6, description="Weight for semantic vector score")
    bm25_weight: float = Field(default=0.4, description="Weight for BM25 score")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    chunk_size: int = Field(default=400, description="Target chunk token count")
    chunk_overlap: int = Field(default=50, description="Chunk token overlap")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model name",
    )
    store_dir: str = Field(default=".khaslana/index", description="Index storage directory")


class SandboxConfig(BaseModel):
    """Sandbox execution limits and settings."""

    enabled: bool = Field(default=True, description="Enable Docker sandbox")
    timeout_seconds: int = Field(default=120, description="Container execution timeout")
    memory_limit: str = Field(default="512m", description="Container memory limit")
    cpu_limit: float = Field(default=1.0, description="Container CPU limit")
    default_image: str = Field(default="python:3.11-slim", description="Default docker image")


class MemoryConfig(BaseModel):
    """Memory and style persistence settings."""

    enabled: bool = Field(
        default=True, description="Enable interaction memory and style adaptation"
    )
    db_path: str = Field(
        default=".khaslana/memory.db", description="SQLite database path for memory"
    )
    recall_limit: int = Field(
        default=3, description="Maximum past memories to inject into prompt"
    )


class KhaslanaConfig(BaseModel):
    """Root configuration model."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    debug: bool = Field(default=False, description="Enable debug logging and tracebacks")
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


def get_user_config_path() -> Path:
    """Return ~/.config/khaslana/config.toml (or platform equivalent)."""
    return Path.home() / ".config" / "khaslana" / "config.toml"


def get_project_config_path(start_path: Path | None = None) -> Path | None:
    """Find .khaslana.toml in current directory or parents."""
    cur = (start_path or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        proj_file = parent / ".khaslana.toml"
        if proj_file.is_file():
            return proj_file
    return None


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    project_dir: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> KhaslanaConfig:
    """Load configuration respecting full precedence hierarchy."""
    merged: dict[str, Any] = {}

    # 1. User config file (~/.config/khaslana/config.toml)
    user_cfg_path = get_user_config_path()
    if user_cfg_path.is_file():
        try:
            with open(user_cfg_path, encoding="utf-8") as f:
                user_data = toml.load(f)
                _deep_update(merged, user_data)
        except Exception:
            pass

    # 2. Project config file (.khaslana.toml)
    proj_cfg_path = get_project_config_path(project_dir)
    if proj_cfg_path and proj_cfg_path.is_file():
        try:
            with open(proj_cfg_path, encoding="utf-8") as f:
                proj_data = toml.load(f)
                _deep_update(merged, proj_data)
        except Exception:
            pass

    # 3. Environment variables
    # Format: KHASLANA_LLM_PROVIDER, KHASLANA_LLM_MODEL, OPENAI_API_KEY, GEMINI_API_KEY, etc.
    env_llm: dict[str, Any] = {}
    if "KHASLANA_LLM_PROVIDER" in os.environ:
        env_llm["provider"] = os.environ["KHASLANA_LLM_PROVIDER"]
    if "KHASLANA_LLM_MODEL" in os.environ:
        env_llm["model"] = os.environ["KHASLANA_LLM_MODEL"]
    if "KHASLANA_OLLAMA_URL" in os.environ:
        env_llm["ollama_url"] = os.environ["KHASLANA_OLLAMA_URL"]
    if "OPENAI_API_KEY" in os.environ:
        env_llm["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    if "GEMINI_API_KEY" in os.environ:
        env_llm["gemini_api_key"] = os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        env_llm["gemini_api_key"] = os.environ["GOOGLE_API_KEY"]

    if env_llm:
        merged.setdefault("llm", {})
        _deep_update(merged["llm"], env_llm)

    if "KHASLANA_DEBUG" in os.environ:
        merged["debug"] = os.environ["KHASLANA_DEBUG"].lower() in {"1", "true", "yes"}

    # 4. CLI overrides
    if cli_overrides:
        _deep_update(merged, cli_overrides)

    return KhaslanaConfig.model_validate(merged)


def save_user_config_value(key_path: str, value: Any) -> Path:
    """Save a setting into ~/.config/khaslana/config.toml."""
    user_cfg_path = get_user_config_path()
    user_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if user_cfg_path.is_file():
        try:
            with open(user_cfg_path, encoding="utf-8") as f:
                data = toml.load(f)
        except Exception:
            data = {}

    keys = key_path.split(".")
    target = data
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = value

    with open(user_cfg_path, "w", encoding="utf-8") as f:
        toml.dump(data, f)

    return user_cfg_path
