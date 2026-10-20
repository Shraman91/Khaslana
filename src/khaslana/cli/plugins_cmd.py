"""CLI commands for managing Khaslana plugins and extensions."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from khaslana.core.config import load_config
from khaslana.core.logging import setup_logging
from khaslana.plugins.loader import PluginLoader
from khaslana.tools.base import ToolRegistry

plugins_app = typer.Typer(
    name="plugins",
    help="Manage Khaslana plugins and dynamic extensions.",
    no_args_is_help=True,
)
console = Console()


@plugins_app.command("list")
def list_plugins(
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """List all discovered plugins, status, and provided tools."""
    config = load_config(cli_overrides={"debug": debug} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    loader = PluginLoader()
    meta_list = loader.get_metadata_list()

    if not meta_list:
        console.print("[dim yellow]No plugins discovered.[/dim yellow]")
        return

    table = Table(title="Discovered Khaslana Plugins", border_style="cyan")
    table.add_column("Plugin", style="bold cyan", min_width=14)
    table.add_column("Version", style="green", width=9)
    table.add_column("Source", style="dim", width=12)
    table.add_column("Status", width=12)
    table.add_column("Provided Tools", style="magenta")
    table.add_column("Description", style="white")

    errors: list[tuple[str, str]] = []

    for m in meta_list:
        if m.error:
            status_str = "[bold red]✗ Error[/bold red]"
            errors.append((m.name, m.error))
        elif m.enabled:
            status_str = "[bold green]✓ Enabled[/bold green]"
        else:
            status_str = "[dim red]○ Disabled[/dim red]"

        tools_str = ", ".join(m.tools) if m.tools else "[dim]None[/dim]"
        table.add_row(
            m.name,
            m.version,
            m.source,
            status_str,
            tools_str,
            m.description or "[dim]No description[/dim]",
        )

    console.print(table)

    if errors:
        err_lines = [f"[bold red]• {name}:[/bold red] {err}" for name, err in errors]
        console.print(
            Panel(
                "\n".join(err_lines),
                title="[bold red]Plugin Load Warnings[/bold red]",
                border_style="red",
            )
        )


@plugins_app.command("enable")
def enable_plugin(
    name: Annotated[str, typer.Argument(help="Name of the plugin to enable")],
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Enable a plugin by name."""
    config = load_config(cli_overrides={"debug": debug} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    loader = PluginLoader()
    plugin = loader.get_plugin(name)
    if not plugin:
        console.print(f"[bold red]Error:[/bold red] Plugin '{name}' was not found.")
        raise typer.Exit(code=1)

    if loader.is_enabled(name):
        console.print(f"[dim yellow]Plugin '{name}' is already enabled.[/dim yellow]")
        return

    loader.enable(name)
    console.print(f"[bold green]✓ Plugin '{name}' enabled successfully.[/bold green]")


@plugins_app.command("disable")
def disable_plugin(
    name: Annotated[str, typer.Argument(help="Name of the plugin to disable")],
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Disable a plugin by name."""
    config = load_config(cli_overrides={"debug": debug} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    loader = PluginLoader()
    plugin = loader.get_plugin(name)
    if not plugin:
        console.print(f"[bold red]Error:[/bold red] Plugin '{name}' was not found.")
        raise typer.Exit(code=1)

    if not loader.is_enabled(name):
        console.print(f"[dim yellow]Plugin '{name}' is already disabled.[/dim yellow]")
        return

    loader.disable(name)
    console.print(f"[bold yellow]✓ Plugin '{name}' disabled.[/bold yellow]")


@plugins_app.command("info")
def info_plugin(
    name: Annotated[str, typer.Argument(help="Name of the plugin to inspect")],
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Show detailed metadata and tool schemas for a plugin."""
    config = load_config(cli_overrides={"debug": debug} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    loader = PluginLoader()
    plugin = loader.get_plugin(name)
    if not plugin:
        console.print(f"[bold red]Error:[/bold red] Plugin '{name}' was not found.")
        raise typer.Exit(code=1)

    meta = plugin.metadata
    if meta.error:
        status_str = f"[bold red]Error: {meta.error}[/bold red]"
    elif meta.enabled:
        status_str = "[bold green]Enabled[/bold green]"
    else:
        status_str = "[dim red]Disabled[/dim red]"

    header_lines = [
        f"[bold]Name:[/bold] {meta.name}",
        f"[bold]Version:[/bold] {meta.version}",
        f"[bold]Author:[/bold] {meta.author or 'Unknown'}",
        f"[bold]Source:[/bold] {meta.source}",
        f"[bold]Status:[/bold] {status_str}",
    ]
    if meta.path:
        header_lines.append(f"[bold]Path:[/bold] {meta.path}")
    if meta.description:
        header_lines.append(f"[bold]Description:[/bold] {meta.description}")

    console.print(Panel("\n".join(header_lines), title=f"Plugin: {meta.name}", border_style="cyan"))

    # Inspect registered tools
    dummy_registry = ToolRegistry()
    try:
        plugin.register(dummy_registry)
        tools = dummy_registry.list_tools()
    except Exception as e:
        console.print(f"[bold red]Failed to inspect tools for '{meta.name}': {e}[/bold red]")
        return

    if not tools:
        console.print("[dim]This plugin does not expose any agent tools.[/dim]")
        return

    console.print(f"\n[bold cyan]Provided Tools ({len(tools)}):[/bold cyan]\n")
    for tool in tools:
        schema: dict[str, Any] = tool.input_schema.model_json_schema()
        properties: dict[str, Any] = schema.get("properties", {})
        required: set[str] = set(schema.get("required", []))

        param_table = Table(
            title=f"Tool: [bold magenta]{tool.name}[/bold magenta] - {tool.description}",
            border_style="magenta",
        )
        param_table.add_column("Parameter", style="bold yellow")
        param_table.add_column("Type", style="cyan")
        param_table.add_column("Required", style="green")
        param_table.add_column("Default", style="dim")
        param_table.add_column("Description", style="white")

        for param_name, prop in properties.items():
            ptype = prop.get("type", "any")
            req_str = "[bold green]Yes[/bold green]" if param_name in required else "[dim]No[/dim]"
            default_val = prop.get("default", "-")
            default_str = str(default_val) if default_val is not None else "-"
            desc = prop.get("description", "")
            param_table.add_row(param_name, ptype, req_str, default_str, desc)

        console.print(param_table)
        console.print()
