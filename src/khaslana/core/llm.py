"""LLM client abstractions, implementations, and fallback router.

Every request sent to any backend is guaranteed to pass through core.redact.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from khaslana.core.config import KhaslanaConfig, LLMConfig
from khaslana.core.errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
)
from khaslana.core.redact import redact


class ChatMessage(BaseModel):
    """Chat message representation."""

    role: Literal["system", "user", "assistant"] = "user"
    content: str


class LLMResponse(BaseModel):
    """LLM completion response metadata."""

    content: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""
        ...

    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a complete chat response."""
        ...

    @abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        """Stream response tokens asynchronously."""
        yield ""

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Convenience method for single prompt generation."""
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        resp = await self.chat(messages)
        return resp.content


class OllamaClient(LLMClient):
    """Ollama local HTTP client."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def list_models(self) -> list[str]:
        """Fetch available models in local Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.config.ollama_url}/api/tags")
                if res.status_code == 200:
                    models = res.json().get("models", [])
                    return [m.get("name", "") for m in models if "name" in m]
                return []
        except Exception:
            return []

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        url = f"{self.config.ollama_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 404:
                    raise LLMError(
                        f"Model '{self.config.model}' not found in Ollama.",
                        hint=f"Run `ollama pull {self.config.model}` first.",
                    )
                res.raise_for_status()
                data = res.json()
                content = data.get("message", {}).get("content", "")
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="ollama",
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                )
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.config.ollama_url}",
                hint="Is Ollama running? Start it with `ollama serve`.",
            ) from e
        except httpx.TimeoutException as e:
            msg = f"Ollama request timed out after {self.config.timeout_seconds}s"
            raise LLMTimeoutError(msg) from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama HTTP error: {e.response.status_code} {e.response.text}") from e

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        url = f"{self.config.ollama_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            async with (
                httpx.AsyncClient(timeout=self.config.timeout_seconds) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                if response.status_code != 200:
                    err_body = await response.aread()
                    raise LLMError(
                        f"Ollama stream error {response.status_code}: {err_body.decode()}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.config.ollama_url}",
                hint="Start Ollama with `ollama serve`",
            ) from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Ollama stream timed out") from e


class OpenAIClient(LLMClient):
    """OpenAI API client using standard chat completions."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def _headers(self) -> dict[str, str]:
        if not self.config.openai_api_key:
            raise LLMAuthError("OpenAI API key missing. Set OPENAI_API_KEY environment variable.")
        return {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        url = f"{self.config.openai_base_url}/chat/completions"
        payload = {
            "model": self.config.model or "gpt-4o-mini",
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                res = await client.post(url, headers=self._headers(), json=payload)
                if res.status_code in (401, 403):
                    raise LLMAuthError("Invalid OpenAI API key")
                res.raise_for_status()
                data = res.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                return LLMResponse(
                    content=choice["message"]["content"] or "",
                    model=data.get("model", self.config.model),
                    provider="openai",
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to reach OpenAI at {url}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("OpenAI request timed out") from e

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        url = f"{self.config.openai_base_url}/chat/completions"
        payload = {
            "model": self.config.model or "gpt-4o-mini",
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        try:
            async with (
                httpx.AsyncClient(timeout=self.config.timeout_seconds) as client,
                client.stream(
                    "POST", url, headers=self._headers(), json=payload
                ) as response,
            ):
                if response.status_code in (401, 403):
                    raise LLMAuthError("Invalid OpenAI API key")
                if response.status_code != 200:
                    err_body = await response.aread()
                    raise LLMError(f"OpenAI error {response.status_code}: {err_body.decode()}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]
        except httpx.ConnectError as e:
            raise LLMConnectionError("Failed to connect to OpenAI endpoint") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("OpenAI stream timed out") from e


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_api_key(self) -> str:
        key = self.config.gemini_api_key
        if not key:
            raise LLMAuthError(
                "Google Gemini API key missing. Set GEMINI_API_KEY environment variable."
            )
        return key

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        system_instruction = None
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_instruction = {"parts": [{"text": m.content}]}
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})
        return system_instruction, contents

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        api_key = self._get_api_key()
        model_name = self.config.model if "gemini" in self.config.model else "gemini-1.5-flash"
        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        url = f"{base_url}/{model_name}:generateContent?key={api_key}"

        system_instruction, contents = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                res = await client.post(url, json=payload)
                if res.status_code in (400, 401, 403):
                    raise LLMAuthError(f"Gemini API error: {res.text}")
                res.raise_for_status()
                data = res.json()
                text = ""
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        text += part.get("text", "")
                return LLMResponse(
                    content=text,
                    model=model_name,
                    provider="gemini",
                )
        except httpx.ConnectError as e:
            raise LLMConnectionError("Failed to connect to Google Gemini endpoint") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Gemini request timed out") from e

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        # For simplicity and robust fallback, fetch chat and yield tokens
        resp = await self.chat(messages)
        for chunk in resp.content.split(" "):
            yield chunk + " "
            await asyncio.sleep(0.01)


class FakeLLM(LLMClient):
    """Deterministic fake LLM for testing without network."""

    def __init__(
        self, config: LLMConfig | None = None, responses: list[str] | None = None
    ) -> None:
        super().__init__(config or LLMConfig(provider="fake", model="fake-model"))
        self.responses = list(responses or ["Hello! I am a fake LLM response."])
        self.call_history: list[list[ChatMessage]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def push_response(self, text: str) -> None:
        self.responses.append(text)

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        self.call_history.append(messages)
        content = self.responses.pop(0) if self.responses else "Default fake answer."
        return LLMResponse(content=content, model="fake-model", provider="fake")

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        self.call_history.append(messages)
        content = self.responses.pop(0) if self.responses else "Default fake answer."
        words = content.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.001)


class LLMRouter:
    """Intelligent router that enforces redaction, retries, and fallback chains."""

    def __init__(
        self,
        config: KhaslanaConfig,
        custom_primary: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.clients: dict[str, LLMClient] = {}

        if custom_primary:
            self.clients[custom_primary.provider_name] = custom_primary
            self.primary_provider = custom_primary.provider_name
        else:
            self.primary_provider = config.llm.provider
            self.clients["ollama"] = OllamaClient(config.llm)
            self.clients["gemini"] = GeminiClient(config.llm)
            self.clients["openai"] = OpenAIClient(config.llm)
            self.clients["fake"] = FakeLLM(config.llm)

    def _sanitize_messages(
        self, messages: Sequence[ChatMessage | dict[str, Any]]
    ) -> list[ChatMessage]:
        """Convert input messages and redact secrets."""
        safe_messages: list[ChatMessage] = []
        for m in messages:
            if isinstance(m, dict):
                r_str = str(m.get("role", "user"))
                role: Literal["system", "user", "assistant"] = (
                    "system"
                    if r_str == "system"
                    else ("assistant" if r_str == "assistant" else "user")
                )
                m = ChatMessage(role=role, content=str(m.get("content", "")))
            redacted_content = redact(m.content)
            safe_messages.append(ChatMessage(role=m.role, content=redacted_content))
        return safe_messages

    def _get_provider_chain(self) -> list[str]:
        """Determine fallback order."""
        chain = [self.primary_provider]
        fallbacks = ["ollama", "gemini", "openai"]
        for f in fallbacks:
            if f not in chain:
                # Only add fallbacks if their credentials/setup are potentially present
                if f == "gemini" and not self.config.llm.gemini_api_key:
                    continue
                if f == "openai" and not self.config.llm.openai_api_key:
                    continue
                chain.append(f)
        return chain

    async def chat(self, messages: Sequence[ChatMessage | dict[str, Any]]) -> LLMResponse:
        """Route chat request through primary with fallback and retries."""
        safe_msgs = self._sanitize_messages(messages)
        chain = self._get_provider_chain()
        last_error: Exception | None = None

        for provider in chain:
            client = self.clients.get(provider)
            if not client:
                continue

            for attempt in range(self.config.llm.max_retries + 1):
                try:
                    return await client.chat(safe_msgs)
                except (LLMConnectionError, LLMTimeoutError) as e:
                    last_error = e
                    if attempt < self.config.llm.max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    break  # Try next provider in fallback chain
                except Exception as e:
                    last_error = e
                    break

        raise LLMError(
            f"All LLM providers in chain {chain} failed.",
            hint=f"Last error: {last_error}",
        )

    async def stream(
        self, messages: Sequence[ChatMessage | dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Route streaming chat request through primary with fallback."""
        safe_msgs = self._sanitize_messages(messages)
        chain = self._get_provider_chain()
        last_error: Exception | None = None

        for provider in chain:
            client = self.clients.get(provider)
            if not client:
                continue

            try:
                # Test connectivity by initiating stream
                stream_gen = client.stream(safe_msgs)
                async for chunk in stream_gen:
                    yield chunk
                return
            except (LLMConnectionError, LLMTimeoutError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        raise LLMError(
            f"All LLM streaming providers in chain {chain} failed.",
            hint=f"Last error: {last_error}",
        )
