"""Data models and interfaces for the Khaslana plugin system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from khaslana.tools.base import ToolRegistry


class PluginMetadata(BaseModel):
    """Metadata describing a discoverable or loaded plugin."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    enabled: bool = True
    tools: list[str] = Field(default_factory=list)
    source: Literal["builtin", "directory", "entrypoint"] = "builtin"
    path: str | None = None
    error: str | None = None


class Plugin(ABC):
    """Base interface that all Khaslana plugins must implement."""

    metadata: PluginMetadata

    @abstractmethod
    def register(self, registry: ToolRegistry) -> None:
        """Register tools and capabilities into the central tool registry."""
        ...
