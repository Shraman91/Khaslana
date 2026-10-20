# Khaslana Plugin System & Extension Guide

Khaslana features a dynamic, isolated plugin architecture designed to extend the AI agent's capabilities with custom tools and external integrations without modifying the core codebase.

---

## Architecture Overview

```
                      ┌────────────────────────────┐
                      │    PluginLoader (Isolated) │
                      └──────────────┬─────────────┘
                                     │
     ┌──────────────────┬────────────┼─────────────────┬──────────────────┐
     ▼                  ▼            ▼                 ▼                  ▼
┌───────────┐    ┌─────────────┐ ┌───────────────┐ ┌───────────────┐ ┌────────────────┐
│ Built-ins │    │ Entry Points│ │ User Config   │ │ Workspace     │ │ Error Sentinel │
│ db-query  │    │ khaslana.   │ │ ~/.config/    │ │ .khaslana/    │ │ (Isolated      │
│ docker    │    │ plugins     │ │ khaslana/     │ │ plugins/      │ │  Import/Init)  │
│ api-tester│    │             │ │ plugins/      │ │               │ │                │
└─────┬─────┘    └──────┬──────┘ └───────┬───────┘ └───────┬───────┘ └────────────────┘
      │                 │                │                 │
      └─────────────────┴────────┬───────┴─────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ ToolRegistry (Agent)  │
                     └───────────────────────┘
```

### Discovery Locations

Khaslana automatically discovers plugins from four sources (in order of priority):
1. **Built-in Plugins**: Core plugins shipped with Khaslana (`db-query`, `docker`, `api-tester`).
2. **Python Entry Points**: External Python packages registering under `khaslana.plugins` in their `pyproject.toml`.
3. **User Global Directory**: `~/.config/khaslana/plugins/` (available across all projects).
4. **Workspace Directory**: `.khaslana/plugins/` (project-specific tools checked into or private to a repo).

### Error Isolation & Safety

- **Import Isolation**: Syntax errors, missing dependencies, or runtime exceptions during a plugin's import do not crash the CLI or other plugins. Khaslana wraps failing modules into a sentinel `_ErrorPlugin` and flags them with an error badge in `khaslana plugins list`.
- **Validation**: Every tool specifies a Pydantic `input_schema`. The AI agent's generated JSON tool calls are validated against this schema before `execute` is invoked.
- **Toggles**: Any plugin can be toggled on or off at runtime via `khaslana plugins enable <name>` or `khaslana plugins disable <name>`, persisted in `~/.config/khaslana/plugins.json`.

---

## Write Your Own Plugin in 5 Minutes

Here is a step-by-step tutorial to create a custom currency converter or utility tool.

### Step 1: Create a Plugin File

In your project repository, create a directory `.khaslana/plugins/` and add `calculator.py`:

```bash
mkdir -p .khaslana/plugins
touch .khaslana/plugins/calculator.py
```

### Step 2: Implement the Tool and Plugin

Add the following code to `.khaslana/plugins/calculator.py`:

```python
"""Custom math expression evaluator plugin."""

from khaslana.plugins.models import Plugin, PluginMetadata
from khaslana.tools.base import Tool, ToolRegistry, ToolResult
from pydantic import BaseModel, Field


class EvalInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate (e.g. '2 ** 10')")


class EvalTool(Tool):
    name: str = "eval_math"
    description: str = "Safely evaluate mathematical expressions using Python's arithmetic engine."
    input_schema: type[BaseModel] = EvalInput

    def execute(self, input: BaseModel) -> ToolResult:
        if not isinstance(input, EvalInput):
            return ToolResult(ok=False, output="", error="Invalid input schema.")

        # Safe arithmetic evaluation
        allowed_chars = set("0123456789+-*/%(). ")
        if not all(c in allowed_chars for c in input.expression):
            return ToolResult(
                ok=False,
                output="",
                error="Disallowed characters in math expression. Only basic arithmetic is permitted.",
            )

        try:
            # Evaluate within clean restricted globals
            result = eval(input.expression, {"__builtins__": {}}, {})  # noqa: S307
            return ToolResult(
                ok=True,
                output=f"Result: {result}",
                metadata={"expression": input.expression, "result": result},
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"Evaluation failed: {e}")


class MathPlugin(Plugin):
    metadata = PluginMetadata(
        name="math-tools",
        version="0.1.0",
        description="Safe arithmetic evaluation tools for the agent.",
        author="Your Name",
        tools=["eval_math"],
    )

    def register(self, registry: ToolRegistry) -> None:
        registry.register(EvalTool())
```

### Step 3: Verify and Inspect

Run `khaslana plugins list`:

```bash
khaslana plugins list
```

You will see `math-tools` listed alongside the built-ins:
```
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Plugin       ┃ Version ┃ Source    ┃ Status    ┃ Provided Tools ┃ Description                 ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ math-tools   │ 0.1.0   │ directory │ ✓ Enabled │ eval_math      │ Safe arithmetic evaluation  │
└──────────────┴─────────┴───────────┴───────────┴────────────────┴─────────────────────────────┘
```

Inspect the parameter schema generated for the agent:

```bash
khaslana plugins info math-tools
```

---

## Alternative: Lightweight `register(registry)` Function

If you don't want to inherit from `Plugin`, you can simply expose a top-level `register(registry)` function in your Python script:

```python
"""Simple standalone plugin module."""

from khaslana.tools.base import Tool, ToolRegistry, ToolResult
from pydantic import BaseModel, Field


class PingInput(BaseModel):
    message: str = Field(default="pong", description="Response text")


class PingTool(Tool):
    name: str = "ping"
    description: str = "Returns ping response."
    input_schema: type[BaseModel] = PingInput

    def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output="pong")


def register(registry: ToolRegistry) -> None:
    """Plugin entry point automatically discovered by Khaslana."""
    registry.register(PingTool())
```

---

## Distributing Plugins via Entry Points

To publish a plugin as a reusable Python package:

1. In your package's `pyproject.toml`, add an entry point under the `khaslana.plugins` group:

```toml
[project.entry-points."khaslana.plugins"]
my_extension = "my_package.plugin:MyPlugin"
```

2. When users `pip install my-package`, Khaslana will automatically discover and load `my_extension`.

---

## Built-in Plugins Reference

| Plugin | Tool Name | Description | Safety Features |
|---|---|---|---|
| **`db-query`** | `db_query` | Read-only SQL query inspection for SQLite databases. | Rejects mutating commands (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc.). Opens connections with `?mode=ro`. |
| **`docker`** | `docker_inspect` | Container listing, configuration inspection, and live log tailing. | Read-only container inspection. Graceful failure if Docker daemon is not active. |
| **`api-tester`**| `api_test` | Send HTTP requests (GET, POST, PUT, etc.) and assert status codes. | Latency measurement, response body truncation, status code assertions. |

---

## CLI Command Reference

- `khaslana plugins list`: Display all discovered plugins, versions, origins, and statuses.
- `khaslana plugins info <name>`: Show plugin metadata, author, and schema parameter tables for all exposed tools.
- `khaslana plugins enable <name>`: Enable a previously disabled plugin.
- `khaslana plugins disable <name>`: Disable a plugin without uninstalling its files.
