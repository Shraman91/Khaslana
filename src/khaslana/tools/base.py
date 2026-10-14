"""Base classes and registry for Khaslana tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class ToolResult(BaseModel):
    """Result of a tool execution."""

    ok: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all tools."""

    name: str
    description: str
    input_schema: type[BaseModel]

    @abstractmethod
    def execute(self, input: BaseModel) -> ToolResult:
        """Execute the tool with the validated input model."""
        ...


class ToolRegistry:
    """Registry maintaining active tools and producing prompt schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_descriptions(self) -> str:
        """Format all registered tools and their schemas for inclusion in prompts."""
        lines = []
        for tool in self._tools.values():
            schema = tool.input_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            params_desc = []
            for prop_name, prop_info in properties.items():
                ptype = prop_info.get("type", "any")
                pdesc = prop_info.get("description", "")
                req_str = "required" if prop_name in required else "optional"
                params_desc.append(f"    - {prop_name} ({ptype}, {req_str}): {pdesc}".rstrip())

            params_str = "\n".join(params_desc) if params_desc else "    (No parameters)"
            tool_entry = (
                f"• Tool: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  Parameters:\n{params_str}"
            )
            lines.append(tool_entry)
        return "\n\n".join(lines)

    def execute(self, name: str, input_data: dict[str, Any] | BaseModel) -> ToolResult:
        """Safely validate arguments and execute the tool."""
        tool = self.get(name)
        if not tool:
            available = list(self._tools.keys())
            return ToolResult(
                ok=False,
                output="",
                error=f"Tool '{name}' is not registered. Available tools: {available}",
            )

        try:
            if isinstance(input_data, BaseModel):
                validated = input_data
            else:
                validated = tool.input_schema.model_validate(input_data)
            return tool.execute(validated)
        except ValidationError as ve:
            return ToolResult(
                ok=False,
                output="",
                error=f"Invalid arguments for tool '{name}': {ve.errors()}",
                metadata={"raw_error": str(ve)},
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                output="",
                error=f"Error executing tool '{name}': {e}",
            )
