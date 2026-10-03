"""Khaslana CLI main entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

from rich.console import Console
from rich.panel import Panel
import typer

from khaslana.cli.config_cmd import config_app
from khaslana.cli.doctor import run_doctor
from khaslana.core.config import load_config
from khaslana.core.errors import KhaslanaError
from khaslana.core.llm import ChatMessage, LLMRouter
from khaslana.core.logging import setup_logging
from khaslana.core.prompts import EXPLAIN_PROMPT_TEMPLATE, SYSTEM_PROMPT

app = typer.Typer(
    name="khaslana",
    help="Khaslana: A production-grade, local-first AI developer tool.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")
console = Console()


def _handle_error(e: Exception, debug: bool) -> None:
    """Format and print errors gracefully without raw tracebacks unless --debug."""
    if isinstance(e, KhaslanaError):
        msg = f"[bold red]Error:[/bold red] {e.message}"
        if e.hint:
            msg += f"\n\n[dim yellow]Hint: {e.hint}[/dim yellow]"
        console.print(Panel(msg, title="[bold red]Khaslana Error[/bold red]", border_style="red"))
    else:
        console.print(Panel(f"[bold red]Unexpected Error:[/bold red] {e}", border_style="red"))
    if debug:
        console.print_exception()


@app.command("ask")
def ask(
    query: Annotated[str, typer.Argument(help="Question or prompt to ask Khaslana")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override LLM model name")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Override LLM provider")] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs and full tracebacks")] = False,
) -> None:
    """Ask Khaslana a technical question or for coding assistance."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    router = LLMRouter(config)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=query),
    ]

    async def _run() -> None:
        try:
            console.print(f"[bold blue]Khaslana ({config.llm.provider}/{config.llm.model}):[/bold blue]")
            async for chunk in router.stream(messages):
                console.print(chunk, end="")
            console.print()
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("explain")
def explain(
    file_path: Annotated[Path, typer.Argument(help="Path to the file to explain")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model name")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Override provider")] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Analyze and explain a code file in detail."""
    if not file_path.is_file():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        console.print(f"[bold red]Error reading file:[/bold red] {e}")
        raise typer.Exit(code=1)

    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    ext = file_path.suffix.lstrip(".") or "plaintext"
    prompt = EXPLAIN_PROMPT_TEMPLATE.format(
        file_path=str(file_path),
        language=ext,
        content=content,
    )

    router = LLMRouter(config)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt),
    ]

    async def _run() -> None:
        try:
            console.print(f"[bold blue]Explaining {file_path.name} ({config.llm.provider}/{config.llm.model}):[/bold blue]")
            async for chunk in router.stream(messages):
                console.print(chunk, end="")
            console.print()
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("doctor")
def doctor(
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Check system dependencies, models, and health."""
    config = load_config(cli_overrides={"debug": debug} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    async def _run() -> None:
        ok = await run_doctor(config, console)
        if not ok:
            raise typer.Exit(code=1)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
