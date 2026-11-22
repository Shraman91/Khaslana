"""Tests for privacy guarantees, single redaction choke point, and secret sanitization."""

from __future__ import annotations

import httpx
import pytest
import respx

from khaslana.core.config import KhaslanaConfig, LLMConfig
from khaslana.core.errors import LLMError
from khaslana.core.llm import (
    ChatMessage,
    FakeLLM,
    LLMRouter,
)
from khaslana.core.redact import redact


class TestPrivacyCloudFallbackGuard:
    @pytest.mark.asyncio
    @respx.mock
    async def test_cloud_fallback_blocked_by_default(self) -> None:
        """When Ollama fails and allow_cloud_fallback is False, router refuses cloud fallback."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        cfg = KhaslanaConfig(
            llm=LLMConfig(
                provider="ollama",
                gemini_api_key="AIzaSyDummyKeyForTestingOnly12345678",
                allow_cloud_fallback=False,
            )
        )
        router = LLMRouter(cfg)

        with pytest.raises(LLMError) as exc_info:
            await router.chat([ChatMessage(role="user", content="hello")])

        err = exc_info.value
        assert "cloud fallback is disabled" in err.message.lower()
        assert "--allow-cloud" in (err.hint or "")

    @pytest.mark.asyncio
    @respx.mock
    async def test_cloud_fallback_allowed_with_flag(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When allow_cloud_fallback is True, router warns and reaches cloud provider."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        ).mock(
            return_value=httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]},
            )
        )

        cfg = KhaslanaConfig(
            llm=LLMConfig(
                provider="ollama",
                gemini_api_key="AIzaSyDummyKeyForTestingOnly12345678",
                allow_cloud_fallback=True,
            )
        )
        router = LLMRouter(cfg)

        resp = await router.chat([ChatMessage(role="user", content="hello")])
        assert resp.content == "Hello from Gemini"
        assert resp.provider == "gemini"

        # Check privacy warning emitted to stderr
        captured = capsys.readouterr()
        assert "Privacy Warning" in captured.err
        assert "gemini" in captured.err.lower()


class TestSingleRedactionChokePoint:
    @pytest.mark.asyncio
    async def test_direct_client_chat_enforces_redaction(self) -> None:
        """Calling LLMClient.chat directly must sanitize secrets without bypassing."""
        fake = FakeLLM(responses=["Echo back"])
        secret_prompt = "My secret token is sk-ant-api03-abcdef1234567890abcdef123456"

        resp = await fake.chat([ChatMessage(role="user", content=secret_prompt)])
        assert resp.content == "Echo back"

        # Check the internal received message was redacted
        received_msg = fake.call_history[0][0].content
        assert "sk-ant-" not in received_msg
        assert "[REDACTED:ANTHROPIC_KEY]" in received_msg

    @pytest.mark.asyncio
    async def test_direct_client_stream_enforces_redaction(self) -> None:
        """Calling LLMClient.stream directly must sanitize secrets without bypassing."""
        fake = FakeLLM(responses=["Streaming response"])
        secret_prompt = "AWS Key AKIAIOSFODNN7EXAMPLE is used here"

        chunks = []
        async for chunk in fake.stream([ChatMessage(role="user", content=secret_prompt)]):
            chunks.append(chunk)

        received_msg = fake.call_history[0][0].content
        assert "AKIAIOSFODNN7EXAMPLE" not in received_msg
        assert "[REDACTED:AWS_ACCESS_KEY]" in received_msg


class TestSecretRedactionCoverage:
    def test_aws_access_keys(self) -> None:
        text = "Keys: AKIA1234567890ABCDEF and ASIA0987654321FEDCBA"
        redacted = redact(text)
        assert "AKIA1234567890ABCDEF" not in redacted
        assert "ASIA0987654321FEDCBA" not in redacted
        assert "[REDACTED:AWS_ACCESS_KEY]" in redacted

    def test_github_tokens(self) -> None:
        classic = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        fine_grained = "github_pat_11ABCDEF_0123456789abcdefghijklmnopqrstuv"
        text = f"Tokens: {classic} and {fine_grained}"
        redacted = redact(text)
        assert classic not in redacted
        assert fine_grained not in redacted
        assert "[REDACTED:GITHUB_PAT]" in redacted
        assert "[REDACTED:GITHUB_FINE_GRAINED]" in redacted

    def test_private_key_blocks(self) -> None:
        rsa_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y123456789abcdefghijklmnopqrstuvwxyz\n"
            "-----END RSA PRIVATE KEY-----"
        )
        openssh_key = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAACFwAAAAdzc2gtcn\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        redacted = redact(f"Server keys:\n{rsa_key}\n{openssh_key}")
        assert "MIIEowIBAAK" not in redacted
        assert "b3BlbnNza" not in redacted
        assert "[REDACTED:PRIVATE_KEY]" in redacted

    def test_env_file_lines(self) -> None:
        env_content = (
            "DATABASE_PASSWORD=SuperSecretPassw0rd!\n"
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
            "JWT_SECRET=super-secret-token-key-2024\n"
            "WEBHOOK_SECRET=whsec_abcdef1234567890\n"
        )
        redacted = redact(env_content)
        assert "SuperSecretPassw0rd!" not in redacted
        assert "wJalrXUtnFEMI" not in redacted
        assert "super-secret-token-key-2024" not in redacted
        assert "whsec_abcdef" not in redacted
        assert "[REDACTED:" in redacted


class TestRedactionFalsePositives:
    def test_git_commit_sha_not_redacted(self) -> None:
        """Git commit SHAs (40-char hex) must NOT be redacted."""
        sha = "c6aa4053d2bb0c4366cb9e49339e1a90cbe43b00"
        text = f"commit {sha} (HEAD -> main)"
        redacted = redact(text)
        assert sha in redacted
        assert "[REDACTED" not in redacted

    def test_uuid_not_redacted(self) -> None:
        """Standard UUIDs must NOT be redacted."""
        uuid_str = "123e4567-e89b-12d3-a456-426614174000"
        text = f"Record id: {uuid_str} loaded successfully"
        redacted = redact(text)
        assert uuid_str in redacted
        assert "[REDACTED" not in redacted

    def test_sha256_hash_not_redacted(self) -> None:
        """64-character SHA-256 content hashes must NOT be redacted."""
        sha256_str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        text = f"File hash: {sha256_str}"
        redacted = redact(text)
        assert sha256_str in redacted
        assert "[REDACTED" not in redacted
