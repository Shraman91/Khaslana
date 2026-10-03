"""CLI command group for configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.syntax import Syntax
import typer

from khaslana.core.config import get_user_config_path, load_config, save_user_config_value

config_app = typer.Typer(name="config", help="View and update Khaslana configuration.")
console = Console()


@config_app.command("show")
def config_show(
    as_json: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show current merged configuration."""
    cfg = load_config()
    cfg_dict = cfg.model_dump()

    # Mask API keys before printing
    if cfg_dict.get("llm", {}).get("openai_api_key"):
        cfg_dict["llm"]["openai_api_key"] = "[CONFIGURED]"
    if cfg_dict.get("llm", {}).get("gemini_api_key"):
        cfg_dict["llm"]["gemini_api_key"] = "[CONFIGURED]"

    formatted_json = json.dumps(cfg_dict, indent=2)
    if as_json:
        console.print(formatted_json)
    else:
        syntax = Syntax(formatted_json, "json", theme="monokai", line_numbers=False)
        console.print("[bold cyan]Current Active Configuration:[/bold cyan]")
        console.print(syntax)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Configuration key path (e.g. llm.model, llm.provider)")],
    value: Annotated[str, typer.Argument(help="Value to set")],
) -> None:
    """Set a configuration value in ~/.config/khaslana/config.toml."""
    # Convert typed values if necessary (bool, int, float)
    parsed_val: bool | int | float | str = value
    if value.lower() in ("true", "false"):
        parsed_val = value.lower() == "true"
    else:
        try:
            parsed_val = int(value)
        except ValueError:
            try:
                parsed_val = float(value)
            except ValueError:
                parsed_val = value

    path = save_user_config_value(key, parsed_val)
    console.print(f"[green]✓[/green] Saved [bold]{key}[/bold] = [cyan]{parsed_val}[/cyan] to {path}")
