"""Secret redaction engine for Khaslana.

Protects against leaking sensitive data to any LLM backend (Ollama, Gemini, OpenAI).
Applies both pattern-based and Shannon entropy-based detection.
"""

from __future__ import annotations

import math
import re
from typing import Pattern

# Explicit pattern matchers for known secret types
KNOWN_PATTERNS: list[tuple[str, Pattern[str]]] = [
    # Private keys
    (
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]+PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
    # OpenAI keys
    ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{20,}\b")),
    # Anthropic keys
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    # Google API keys
    ("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,45}\b")),
    # AWS Access Key ID
    ("AWS_ACCESS_KEY", re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b")),
    # GitHub tokens
    ("GITHUB_PAT", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,255}\b")),
    ("GITHUB_FINE_GRAINED", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,255}\b")),
    # Slack tokens
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,80}\b")),
    # JWT tokens
    (
        "JWT_TOKEN",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}\b"),
    ),
    # Connection strings (Postgres, MySQL, MongoDB, Redis, etc.)
    (
        "CONNECTION_URI",
        re.compile(
            r"(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://([^:]+):([^@\s]+)@([^\s/:]+)(?::\d+)?(?:/[^\s]*)?",
            re.IGNORECASE,
        ),
    ),
    # Generic key/secret assignment: KEY=value or "key": "value"
    (
        "KEY_VALUE_SECRET",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|passwd|auth[_-]?token|access[_-]?token|private[_-]?key|client[_-]?secret)\b\s*[:=]\s*['\"]?([A-Za-z0-9_!#$%&*+\-/=?^`{|}~.]{8,})['\"]?",
        ),
    ),
    # Bearer tokens
    (
        "BEARER_TOKEN",
        re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-\.]{20,})\b"),
    ),
]


def shannon_entropy(data: str) -> float:
    """Calculate the Shannon entropy of a string."""
    if not data:
        return 0.0
    entropy = 0.0
    length = len(data)
    frequencies = {char: data.count(char) for char in set(data)}
    for count in frequencies.values():
        p_x = count / length
        entropy += -p_x * math.log2(p_x)
    return entropy


def is_high_entropy(token: str, min_len: int = 24) -> bool:
    """Check if an isolated token appears to be high entropy (random secret / token)."""
    if len(token) < min_len:
        return False
    # Ignore tokens with path separators
    if "/" in token or "\\" in token or ":" in token:
        return False
    # Check if only alphanumeric + common base64 chars
    if not re.match(r"^[A-Za-z0-9+/=_-]+$", token):
        return False

    entropy = shannon_entropy(token)
    # Hex tokens (0-9a-f) have a maximum theoretical entropy of log2(16) = 4.0.
    # Base64/alphanumeric have higher max theoretical entropy.
    is_hex = bool(re.match(r"^[0-9a-fA-F]+$", token))
    min_threshold = 3.0 if is_hex else 3.8

    return entropy >= min_threshold


def redact(text: str) -> str:
    """Redact all sensitive information from the input text.

    Replaces discovered secrets with [REDACTED:<TYPE>] or [REDACTED:SECRET].
    """
    if not text:
        return text

    redacted = text

    # 1. Connection URIs specifically (mask the password part)
    def mask_uri(match: re.Match[str]) -> str:
        full = match.group(0)
        user = match.group(1)
        pwd = match.group(2)
        return full.replace(f"{user}:{pwd}@", f"{user}:[REDACTED:PASSWORD]@")

    conn_pattern = dict(KNOWN_PATTERNS)["CONNECTION_URI"]
    redacted = conn_pattern.sub(mask_uri, redacted)

    # 2. Key-value secret assignments
    def mask_kv(match: re.Match[str]) -> str:
        key_name = match.group(1)
        val = match.group(2)
        if val.lower() in {"none", "false", "true", "null", "undefined", "changeme", "example", "xxx"}:
            return match.group(0)
        return match.group(0).replace(val, f"[REDACTED:{key_name.upper()}]")

    kv_pattern = dict(KNOWN_PATTERNS)["KEY_VALUE_SECRET"]
    redacted = kv_pattern.sub(mask_kv, redacted)

    # 3. Bearer tokens
    def mask_bearer(match: re.Match[str]) -> str:
        token = match.group(1)
        return f"Bearer [REDACTED:BEARER_TOKEN]"

    bearer_pattern = dict(KNOWN_PATTERNS)["BEARER_TOKEN"]
    redacted = bearer_pattern.sub(mask_bearer, redacted)

    # 4. Other specific known patterns
    for name, pattern in KNOWN_PATTERNS:
        if name in ("CONNECTION_URI", "KEY_VALUE_SECRET", "BEARER_TOKEN"):
            continue
        redacted = pattern.sub(f"[REDACTED:{name}]", redacted)

    # 5. Token-level high-entropy scanning for uncaptured secrets
    words = re.findall(r"\b[A-Za-z0-9+/=_-]{24,}\b", redacted)
    for word in set(words):
        if "[REDACTED" in word:
            continue
        if is_high_entropy(word):
            redacted = redacted.replace(word, "[REDACTED:HIGH_ENTROPY_SECRET]")

    return redacted
