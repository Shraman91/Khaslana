"""Utility helpers for text formatting and cryptographic functions."""

import hashlib
import json
from typing import Any


def format_json_response(data: dict[str, Any]) -> str:
    """Format dictionary to pretty JSON."""
    return json.dumps(data, indent=2)


def compute_content_hash(text: str) -> str:
    """Calculate SHA256 hex digest of string content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
