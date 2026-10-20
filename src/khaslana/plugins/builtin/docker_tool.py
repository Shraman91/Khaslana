"""Built-in docker plugin: container inspection and log viewing."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from khaslana.plugins.models import Plugin, PluginMetadata
from khaslana.tools.base import Tool, ToolRegistry, ToolResult


class DockerToolInput(BaseModel):
    """Input parameters for the docker_inspect tool."""

    action: Literal["list", "logs", "inspect"] = Field(
        description="Action to perform: 'list', 'logs', or 'inspect'"
    )
    container: str | None = Field(
        default=None,
        description="Container ID or name (required for 'logs' and 'inspect')",
    )
    tail: int = Field(
        default=50,
        description="Number of recent log lines to retrieve when action='logs'",
    )


class DockerTool(Tool):
    """Inspects running containers and fetches logs via Docker SDK."""

    name: str = "docker_inspect"
    description: str = (
        "Interact with Docker containers: list containers, inspect configuration, "
        "or stream recent logs. Requires a running Docker daemon."
    )
    input_schema: type[BaseModel] = DockerToolInput

    def execute(self, input: BaseModel) -> ToolResult:
        if not isinstance(input, DockerToolInput):
            return ToolResult(ok=False, output="", error="Invalid input model for docker_inspect")

        try:
            import docker  # type: ignore[import-untyped]

            client = docker.from_env()
            client.ping()
        except Exception as e:
            return ToolResult(
                ok=False,
                output="",
                error=f"Docker is unavailable or daemon is not running: {e}",
            )

        try:
            if input.action == "list":
                containers = client.containers.list(all=True)
                if not containers:
                    return ToolResult(ok=True, output="No Docker containers found.")

                lines = ["| ID | Image | Status | Name |", "| --- | --- | --- | --- |"]
                for c in containers:
                    c_id = c.short_id
                    img = str(c.image.tags[0]) if c.image.tags else c.image.short_id
                    status = c.status
                    name = c.name
                    lines.append(f"| {c_id} | {img} | {status} | {name} |")

                return ToolResult(
                    ok=True,
                    output="\n".join(lines),
                    metadata={"container_count": len(containers)},
                )

            if not input.container:
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"Parameter 'container' is required for action '{input.action}'.",
                )

            container = client.containers.get(input.container)

            if input.action == "logs":
                raw_logs = container.logs(tail=input.tail)
                decoded = (
                    raw_logs.decode("utf-8", errors="replace")
                    if isinstance(raw_logs, bytes)
                    else str(raw_logs)
                )
                return ToolResult(
                    ok=True,
                    output=decoded or "(No log output recorded)",
                    metadata={"container": input.container, "tail": input.tail},
                )

            if input.action == "inspect":
                attrs = {
                    "id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "image": str(container.image),
                    "created": container.attrs.get("Created"),
                    "state": container.attrs.get("State", {}),
                    "network_settings": container.attrs.get("NetworkSettings", {}).get(
                        "Ports", {}
                    ),
                }
                return ToolResult(
                    ok=True,
                    output=json.dumps(attrs, indent=2),
                    metadata={"container": input.container},
                )

            return ToolResult(
                ok=False,
                output="",
                error=f"Unknown action: {input.action}",
            )

        except Exception as e:
            return ToolResult(
                ok=False,
                output="",
                error=f"Docker operation failed: {e}",
            )


class DockerPlugin(Plugin):
    """Plugin providing Docker container inspection capabilities."""

    metadata = PluginMetadata(
        name="docker",
        version="0.1.0",
        description="Docker container listing, logs, and inspect tool",
        author="Khaslana Team",
        enabled=True,
        tools=["docker_inspect"],
        source="builtin",
    )

    def register(self, registry: ToolRegistry) -> None:
        registry.register(DockerTool())
