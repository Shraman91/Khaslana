"""Pydantic models for the Khaslana Memory System and Style Profile."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRecord(BaseModel):
    """Record of a user query and LLM answer."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    query_text: str
    response_text: str
    timestamp: float = Field(default_factory=time.time)
    session_id: str | None = None
    model: str | None = None
    provider: str | None = None
    tokens_used: int | None = None


class FixRecord(BaseModel):
    """Record of an accepted or rejected code fix."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    file_path: str
    original_code: str
    fixed_code: str
    diff: str
    status: Literal["accepted", "rejected"]
    test_cmd: str | None = None
    attempts_count: int = 1
    timestamp: float = Field(default_factory=time.time)


class FeedbackRecord(BaseModel):
    """User feedback rating for an interaction."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    interaction_id: str
    interaction_type: Literal["query", "fix"]
    rating: Literal["good", "bad"]
    notes: str | None = None
    timestamp: float = Field(default_factory=time.time)


class StyleRule(BaseModel):
    """A single detected coding style rule or pattern."""

    name: str
    description: str
    pattern_type: Literal["naming", "typing", "docstring", "framework", "general"]
    confidence: float = 1.0
    example: str | None = None


class StyleProfile(BaseModel):
    """Compact coding style profile learned from codebase and user interactions."""

    naming_convention: str = "snake_case functions/variables, PascalCase classes"
    typing_discipline: str = "Type hints on function arguments and return values"
    docstring_style: str = "Descriptive docstrings for modules, classes, and public functions"
    framework_patterns: list[str] = Field(default_factory=list)
    rules: list[StyleRule] = Field(default_factory=list)
    last_updated: float = Field(default_factory=time.time)

    def to_prompt_context(self) -> str:
        """Format style profile into a compact prompt injection block."""
        lines = [
            "### User Coding Style Profile",
            f"- Naming: {self.naming_convention}",
            f"- Typing: {self.typing_discipline}",
            f"- Docstrings: {self.docstring_style}",
        ]
        if self.framework_patterns:
            patterns_str = ", ".join(self.framework_patterns)
            lines.append(f"- Framework Idioms: {patterns_str}")
        if self.rules:
            lines.append("- Specific Rules:")
            for r in self.rules[:5]:
                lines.append(f"  * {r.name}: {r.description}")
        return "\n".join(lines)


class MemorySearchResult(BaseModel):
    """Result of a semantic search over past memory interactions."""

    record_id: str
    record_type: Literal["query", "fix"]
    snippet: str
    similarity: float
    timestamp: float
    metadata: dict[str, Any] = Field(default_factory=dict)
