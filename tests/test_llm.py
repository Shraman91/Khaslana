"""Tests for core/llm.py — LLM clients and router."""

from __future__ import annotations

import pytest
import respx
import httpx

from khaslana.core.config import KhaslanaConfig, LLMConfig
from khaslana.core.errors import LLMAuthError, LLMConnectionError, LLMError
from khaslana.core.llm import (
    ChatMessage,
    FakeLLM,
    LLMResponse,
    LLMRouter,
    OllamaClient,
)


@pytest.fixture()
def base_config() -> KhaslanaConfig:
    return KhaslanaConfig(
        llm=LLMConfig(provider="fake", model="fake-model", ollama_url="http://localhost:11434")
    )


@pytest.fixture()
def ollama_config() -> KhaslanaConfig:
    return KhaslanaConfig(
        llm=LLMConfig(provider="ollama", model="llama3", ollama_url="http://localhost:11434")
    )


class TestFakeLLM:
    @pytest.mark.asyncio
    async def test_chat_returns_response(self) -> None:
        llm = FakeLLM(responses=["Hello world!"])
        msgs = [ChatMessage(role="user", content="hi")]
        resp = await llm.chat(msgs)
        assert resp.content == "Hello world!"
        assert resp.provider == "fake"

    @pytest.mark.asyncio
    async def test_chat_records_history(self) -> None:
        llm = FakeLLM(responses=["first", "second"])
        msgs = [ChatMessage(role="user", content="test")]
        await llm.chat(msgs)
        await llm.chat(msgs)
        assert len(llm.call_history) == 2

    @pytest.mark.asyncio
    async def test_stream_returns_chunks(self) -> None:
        llm = FakeLLM(responses=["Hello world"])
        msgs = [ChatMessage(role="user", content="hi")]
        chunks: list[str] = []
        async for chunk in llm.stream(msgs):
            chunks.append(chunk)
        full = "".join(chunks).strip()
        assert "Hello" in full
        assert "world" in full

    @pytest.mark.asyncio
    async def test_default_response_when_empty(self) -> None:
        llm = FakeLLM(responses=[])
        msgs = [ChatMessage(role="user", content="test")]
        resp = await llm.chat(msgs)
        assert "fake" in resp.content.lower() or len(resp.content) > 0


class TestOllamaClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_chat_success(self, ollama_config: KhaslanaConfig) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "Hi there!"}, "prompt_eval_count": 5, "eval_count": 3},
            )
        )
        client = OllamaClient(ollama_config.llm)
        msgs = [ChatMessage(role="user", content="hello")]
        resp = await client.chat(msgs)
        assert resp.content == "Hi there!"
        assert resp.provider == "ollama"

    @pytest.mark.asyncio
    @respx.mock
    async def test_chat_connection_error(self, ollama_config: KhaslanaConfig) -> None:
        respx.post("http://localhost:11434/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        client = OllamaClient(ollama_config.llm)
        msgs = [ChatMessage(role="user", content="hello")]
        with pytest.raises(LLMConnectionError):
            await client.chat(msgs)

    @pytest.mark.asyncio
    @respx.mock
    async def test_chat_model_not_found(self, ollama_config: KhaslanaConfig) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )
        client = OllamaClient(ollama_config.llm)
        msgs = [ChatMessage(role="user", content="hello")]
        with pytest.raises(LLMError):
            await client.chat(msgs)

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_models_success(self, ollama_config: KhaslanaConfig) -> None:
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200, json={"models": [{"name": "llama3"}, {"name": "mistral"}]}
            )
        )
        client = OllamaClient(ollama_config.llm)
        models = await client.list_models()
        assert "llama3" in models
        assert "mistral" in models

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_models_failure_returns_empty(self, ollama_config: KhaslanaConfig) -> None:
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        client = OllamaClient(ollama_config.llm)
        models = await client.list_models()
        assert models == []


class TestLLMRouter:
    @pytest.mark.asyncio
    async def test_uses_custom_primary(self, base_config: KhaslanaConfig) -> None:
        fake = FakeLLM(responses=["Routed!"])
        router = LLMRouter(base_config, custom_primary=fake)
        resp = await router.chat([ChatMessage(role="user", content="test")])
        assert resp.content == "Routed!"

    @pytest.mark.asyncio
    async def test_redacts_secrets_before_sending(self, base_config: KhaslanaConfig) -> None:
        """Ensure secrets in user messages are stripped before reaching the LLM."""
        fake = FakeLLM(responses=["ok"])
        router = LLMRouter(base_config, custom_primary=fake)
        secret_query = "My key is sk-proj-SECRETSECRETSECRETSECRET123456789"
        await router.chat([ChatMessage(role="user", content=secret_query)])

        # The fake LLM's recorded call history should have the content redacted
        received = fake.call_history[0][0].content
        assert "sk-proj-" not in received
        assert "[REDACTED" in received

    @pytest.mark.asyncio
    async def test_stream_yields_text(self, base_config: KhaslanaConfig) -> None:
        fake = FakeLLM(responses=["Hello streaming world"])
        router = LLMRouter(base_config, custom_primary=fake)
        chunks: list[str] = []
        async for chunk in router.stream([ChatMessage(role="user", content="hi")]):
            chunks.append(chunk)
        result = "".join(chunks)
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_stream_redacts_secrets(self, base_config: KhaslanaConfig) -> None:
        fake = FakeLLM(responses=["ok"])
        router = LLMRouter(base_config, custom_primary=fake)
        async for _ in router.stream([ChatMessage(role="user", content="key=sk-proj-SECRETSECRETSECRETSECRET123456789")]):
            pass
        received = fake.call_history[0][0].content
        assert "sk-proj-" not in received


class TestChatMessageModel:
    def test_default_role_is_user(self) -> None:
        msg = ChatMessage(content="test")
        assert msg.role == "user"

    def test_system_role(self) -> None:
        msg = ChatMessage(role="system", content="You are a helper.")
        assert msg.role == "system"

    def test_model_dump_as_dict(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        d = msg.model_dump()
        assert d["role"] == "user"
        assert d["content"] == "hello"
