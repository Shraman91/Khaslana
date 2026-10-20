"""Built-in api-tester plugin: HTTP request verification and assertions."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from khaslana.plugins.models import Plugin, PluginMetadata
from khaslana.tools.base import Tool, ToolRegistry, ToolResult


class APITesterInput(BaseModel):
    """Input parameters for the api_test tool."""

    url: str = Field(description="Target HTTP or HTTPS endpoint URL")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"] = Field(
        default="GET", description="HTTP request method"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Custom HTTP headers to send with the request"
    )
    body: dict[str, Any] | str | None = Field(
        default=None, description="Request body payload (JSON dictionary or raw string)"
    )
    expected_status: int = Field(
        default=200, description="Expected HTTP response status code"
    )
    timeout_seconds: float = Field(
        default=10.0, description="Request timeout limit in seconds"
    )


class APITesterTool(Tool):
    """Executes HTTP requests and asserts response status codes and performance."""

    name: str = "api_test"
    description: str = (
        "Send an HTTP request to an API endpoint and verify the response status code, "
        "headers, and response body structure."
    )
    input_schema: type[BaseModel] = APITesterInput

    def execute(self, input: BaseModel) -> ToolResult:
        if not isinstance(input, APITesterInput):
            return ToolResult(ok=False, output="", error="Invalid input model for api_test")

        start_time = time.perf_counter()
        req_kwargs: dict[str, Any] = {
            "method": input.method,
            "url": input.url,
            "headers": input.headers,
            "timeout": input.timeout_seconds,
        }

        if input.body is not None:
            if isinstance(input.body, dict):
                req_kwargs["json"] = input.body
            else:
                req_kwargs["content"] = str(input.body)

        try:
            with httpx.Client(follow_redirects=True) as client:
                response = client.request(**req_kwargs)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            status_ok = response.status_code == input.expected_status
            body_text = response.text
            try:
                # Try pretty formatting if JSON
                parsed_json = response.json()
                body_formatted = json.dumps(parsed_json, indent=2)
                if len(body_formatted) > 2000:
                    body_formatted = body_formatted[:2000] + "\n...(truncated)"
            except Exception:
                body_formatted = (
                    body_text[:2000] + "\n...(truncated)" if len(body_text) > 2000 else body_text
                )

            report = (
                f"URL: {input.method} {input.url}\n"
                f"Status: {response.status_code} (Expected: {input.expected_status})\n"
                f"Latency: {latency_ms}ms\n"
                f"Content-Type: {response.headers.get('content-type', 'unknown')}\n\n"
                f"Response Body:\n{body_formatted}"
            )

            if status_ok:
                return ToolResult(
                    ok=True,
                    output=report,
                    metadata={
                        "status_code": response.status_code,
                        "latency_ms": latency_ms,
                        "expected_status": input.expected_status,
                    },
                )

            return ToolResult(
                ok=False,
                output=report,
                error=(
                    f"Assertion failed: expected HTTP {input.expected_status}, "
                    f"got HTTP {response.status_code} ({latency_ms}ms)"
                ),
                metadata={
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                    "expected_status": input.expected_status,
                },
            )

        except Exception as e:
            return ToolResult(
                ok=False,
                output="",
                error=f"API request failed: {e}",
            )


class APITesterPlugin(Plugin):
    """Plugin providing HTTP API testing capabilities."""

    metadata = PluginMetadata(
        name="api-tester",
        version="0.1.0",
        description="HTTP request verification tool with status assertions",
        author="Khaslana Team",
        enabled=True,
        tools=["api_test"],
        source="builtin",
    )

    def register(self, registry: ToolRegistry) -> None:
        registry.register(APITesterTool())
