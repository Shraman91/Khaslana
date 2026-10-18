"""CLI command groups for memory management and user feedback."""

from __future__ import annotations

import contextlib
import datetime
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from khaslana.core.config import load_config
from khaslana.memory.store import MemoryStore
from khaslana.memory.style import StyleLearner
from khaslana.retrieval.embeddings import get_embedding_provider

memory_app = typer.Typer(name="memory", help="Manage interaction memory and style profile.")
feedback_app = typer.Typer(name="feedback", help="Provide feedback on past responses or fixes.")
console = Console()


def _get_memory_store() -> MemoryStore:
    """Helper to initialize the MemoryStore from active config."""
    config = load_config()
    db_path = Path.cwd() / config.memory.db_path
    embedder = None
    with contextlib.suppress(Exception):
        embedder = get_embedding_provider(config)
    return MemoryStore(db_path=db_path, embedding_provider=embedder)


@memory_app.command("list")
def memory_list(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum items to display")] = 15,
) -> None:
    """List recent queries and code fixes recorded in local memory."""
    store = _get_memory_store()
    queries = store.list_queries(limit=limit)
    fixes = store.list_fixes(limit=limit)

    if not queries and not fixes:
        console.print("[dim yellow]No recorded interactions in memory yet.[/dim yellow]")
        return

    table = Table(title="Khaslana Interaction Memory", border_style="cyan")
    table.add_column("ID", style="bold green", width=12)
    table.add_column("Type", style="magenta", width=8)
    table.add_column("Timestamp", style="dim", width=19)
    table.add_column("Summary / Snippet", style="white")

    for q in queries:
        dt = datetime.datetime.fromtimestamp(q.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        snippet = f"Q: {q.query_text[:60]}... \n↳ {q.response_text[:80]}..."
        table.add_row(q.id, "Query", dt, snippet)

    for f in fixes:
        dt = datetime.datetime.fromtimestamp(f.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        snippet = f"Fix ({f.status}) for {f.file_path} (attempts: {f.attempts_count})"
        table.add_row(f.id, "Fix", dt, snippet)

    console.print(table)


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Search query or keyword")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results")] = 5,
) -> None:
    """Search interaction history using semantic recall."""
    store = _get_memory_store()
    results = store.search_interactions(query, limit=limit)

    if not results:
        console.print(f"[dim yellow]No memory records matched '{query}'.[/dim yellow]")
        return

    table = Table(title=f"Memory Search Results: '{query}'", border_style="blue")
    table.add_column("ID", style="bold green", width=12)
    table.add_column("Type", style="magenta", width=8)
    table.add_column("Similarity", style="cyan", width=10)
    table.add_column("Snippet", style="white")

    for r in results:
        table.add_row(r.record_id, r.record_type.capitalize(), f"{r.similarity:.2f}", r.snippet)

    console.print(table)


@memory_app.command("forget")
def memory_forget(
    record_id: Annotated[str, typer.Argument(help="ID of the interaction record to delete")],
) -> None:
    """Delete a specific interaction record from memory."""
    store = _get_memory_store()
    success = store.forget(record_id)
    if success:
        console.print(f"[bold green]✓ Memory record '{record_id}' removed.[/bold green]")
    else:
        console.print(f"[yellow]Record '{record_id}' not found in memory store.[/yellow]")


@memory_app.command("clear")
def memory_clear(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Clear all interaction history, fixes, and learned style profiles."""
    if not yes:
        confirmed = typer.confirm(
            "Clear all local memory and learned style profiles?", default=False
        )
        if not confirmed:
            console.print("[dim]Operation aborted. Memory intact.[/dim]")
            return

    store = _get_memory_store()
    store.clear()
    console.print("[bold green]✓ All local memories and style profiles cleared.[/bold green]")


@memory_app.command("export")
def memory_export(
    out: Annotated[Path | None, typer.Option("--out", "-o", help="File path to save JSON")] = None,
) -> None:
    """Export all interaction memory and style profiles as JSON."""
    store = _get_memory_store()
    data = store.export_data()
    json_text = json.dumps(data, indent=2)

    if out:
        out.write_text(json_text, encoding="utf-8")
        console.print(f"[bold green]✓ Memory exported to {out}[/bold green]")
    else:
        syntax = Syntax(json_text, "json", theme="monokai", line_numbers=False)
        console.print(syntax)


@memory_app.command("profile")
def memory_profile() -> None:
    """Show the currently learned coding style profile."""
    store = _get_memory_store()
    profile = store.load_style_profile()
    if not profile:
        # Analyze codebase now
        learner = StyleLearner()
        profile = learner.analyze_codebase(Path.cwd())
        store.save_style_profile(profile)

    panel_content = profile.to_prompt_context()
    console.print(
        Panel(panel_content, title="Active Coding Style Profile", border_style="magenta")
    )


@feedback_app.command("good")
def feedback_good(
    interaction_id: Annotated[
        str | None,
        typer.Argument(help="ID of interaction to rate (defaults to most recent)"),
    ] = None,
    notes: Annotated[str | None, typer.Option("--notes", "-n", help="Optional comment")] = None,
) -> None:
    """Mark an interaction as helpful / correct."""
    store = _get_memory_store()
    target_id = interaction_id or store.get_latest_interaction_id()
    if not target_id:
        console.print("[yellow]No recent interactions found to provide feedback on.[/yellow]")
        raise typer.Exit(code=1)

    store.record_feedback(
        interaction_id=target_id,
        interaction_type="query",
        rating="good",
        notes=notes,
    )
    console.print(
        f"[bold green]✓ Recorded feedback: GOOD for interaction '{target_id}'[/bold green]"
    )


@feedback_app.command("bad")
def feedback_bad(
    interaction_id: Annotated[
        str | None,
        typer.Argument(help="ID of interaction to rate (defaults to most recent)"),
    ] = None,
    notes: Annotated[str | None, typer.Option("--notes", "-n", help="Optional comment")] = None,
) -> None:
    """Mark an interaction as unhelpful / incorrect."""
    store = _get_memory_store()
    target_id = interaction_id or store.get_latest_interaction_id()
    if not target_id:
        console.print("[yellow]No recent interactions found to provide feedback on.[/yellow]")
        raise typer.Exit(code=1)

    store.record_feedback(
        interaction_id=target_id,
        interaction_type="query",
        rating="bad",
        notes=notes,
    )
    console.print(
        f"[bold yellow]✓ Recorded feedback: BAD for interaction '{target_id}'[/bold yellow]"
    )
