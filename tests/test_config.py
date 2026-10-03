"""Tests for core/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from khaslana.core.config import KhaslanaConfig, LLMConfig, _deep_update, load_config


class TestDeepUpdate:
    def test_simple_update(self) -> None:
        base = {"a": 1, "b": 2}
        _deep_update(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_update(self) -> None:
        base = {"llm": {"model": "llama3", "provider": "ollama"}}
        _deep_update(base, {"llm": {"model": "mistral"}})
        assert base == {"llm": {"model": "mistral", "provider": "ollama"}}

    def test_deep_nesting(self) -> None:
        base = {"a": {"b": {"c": 1}}}
        _deep_update(base, {"a": {"b": {"d": 2}}})
        assert base["a"]["b"] == {"c": 1, "d": 2}


class TestKhaslanaConfig:
    def test_defaults(self) -> None:
        cfg = KhaslanaConfig()
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.model == "llama3"
        assert cfg.debug is False

    def test_llm_config_defaults(self) -> None:
        llm = LLMConfig()
        assert llm.ollama_url == "http://localhost:11434"
        assert llm.temperature == 0.2
        assert llm.max_tokens == 4096


class TestLoadConfig:
    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KHASLANA_LLM_MODEL", "gemma3")
        monkeypatch.setenv("KHASLANA_LLM_PROVIDER", "ollama")
        cfg = load_config()
        assert cfg.llm.model == "gemma3"
        assert cfg.llm.provider == "ollama"

    def test_openai_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-here")
        cfg = load_config()
        assert cfg.llm.openai_api_key == "sk-test-key-here"

    def test_gemini_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestGeminiKey")
        cfg = load_config()
        assert cfg.llm.gemini_api_key == "AIzaTestGeminiKey"

    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KHASLANA_LLM_MODEL", "from_env")
        cfg = load_config(cli_overrides={"llm": {"model": "from_cli"}})
        assert cfg.llm.model == "from_cli"

    def test_debug_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KHASLANA_DEBUG", "true")
        cfg = load_config()
        assert cfg.debug is True

    def test_project_config_file(self, tmp_path: Path) -> None:
        proj_toml = tmp_path / ".khaslana.toml"
        proj_toml.write_text('[llm]\nmodel = "from_project_file"\n')
        cfg = load_config(project_dir=tmp_path)
        assert cfg.llm.model == "from_project_file"

    def test_cli_override_beats_project_file(self, tmp_path: Path) -> None:
        proj_toml = tmp_path / ".khaslana.toml"
        proj_toml.write_text('[llm]\nmodel = "from_project_file"\n')
        cfg = load_config(project_dir=tmp_path, cli_overrides={"llm": {"model": "from_cli"}})
        assert cfg.llm.model == "from_cli"
