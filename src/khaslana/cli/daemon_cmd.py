"""CLI commands for background indexing, live file watching, and daemon lifecycle."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from khaslana.core.config import load_config
from khaslana.core.errors import DaemonError
from khaslana.core.logging import setup_logging
from khaslana.intelligence.symbols import SymbolTable
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import get_embedding_provider
from khaslana.retrieval.models import IncrementalSummary
from khaslana.retrieval.store import ChromaVectorStore, MemoryVectorStore
from khaslana.retrieval.watcher import DaemonManager, IncrementalUpdater, RepoWatcher

console = Console()
daemon_app = typer.Typer(help="Manage background indexing daemon", no_args_is_help=True)


@daemon_app.command("start")
def daemon_start(
    path: Annotated[Path, typer.Argument(help="Project directory to watch")] = Path("."),
) -> None:
    """Start background indexing daemon."""
    manager = DaemonManager(path)
    try:
        pid = manager.start()
        console.print(
            Panel(
                f"[bold green]✓ Khaslana background indexing daemon started![/bold green]\n\n"
                f"[bold]PID:[/bold] {pid}\n"
                f"[bold]Watched Path:[/bold] {manager.root_dir}\n"
                f"[bold]Log File:[/bold] {manager.log_file}\n\n"
                "[dim]Run `khaslana daemon status` to inspect or "
                "`khaslana daemon stop` to terminate.[/dim]",
                title="Daemon Started",
                border_style="green",
            )

        )
    except DaemonError as e:
        console.print(f"[bold yellow]Daemon error:[/bold yellow] {e}")
        raise typer.Exit(1) from e


@daemon_app.command("stop")
def daemon_stop(
    path: Annotated[Path, typer.Argument(help="Project directory to check")] = Path("."),
) -> None:
    """Stop running indexing daemon."""
    manager = DaemonManager(path)
    try:
        pid = manager.stop()
        console.print(f"[green]✓ Indexing daemon (PID {pid}) stopped.[/green]")
    except DaemonError as e:
        console.print(f"[yellow]{e}[/yellow]")


@daemon_app.command("status")
def daemon_status(
    path: Annotated[Path, typer.Argument(help="Project directory to check")] = Path("."),
) -> None:
    """Check current status of the background indexing daemon."""
    manager = DaemonManager(path)
    st = manager.status()
    if st.get("running"):
        console.print(
            Panel(
                f"[bold green]● Daemon Status: RUNNING[/bold green]\n\n"
                f"[bold]PID:[/bold] {st['pid']}\n"
                f"[bold]Uptime:[/bold] {st['uptime_seconds']}s\n"
                f"[bold]Watched Directory:[/bold] {st['root_dir']}\n"
                f"[bold]Log:[/bold] {st['log_file']}",
                title="Daemon Status",
                border_style="green",
            )
        )
    else:
        stale_note = " (stale PID file cleaned up)" if st.get("stale_cleaned") else ""
        console.print(f"[dim]● Daemon Status: STOPPED{stale_note}[/dim]")


def watch_cmd(
    path: Annotated[Path, typer.Argument(help="Project directory to watch")] = Path("."),
    debounce: Annotated[
        float, typer.Option("--debounce", help="Debounce seconds before re-indexing")
    ] = 0.5,
    daemon_mode: Annotated[
        bool, typer.Option("--daemon-mode", hidden=True, help="Internal daemon worker flag")
    ] = False,
    memory: Annotated[
        bool, typer.Option("--memory", help="Use in-memory vector store (ephemeral)")
    ] = False,
) -> None:
    """Watch codebase files and incrementally update the search index in real time."""
    root = path.resolve()
    config = load_config()
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    persist_dir = root / config.retrieval.store_dir
    store = MemoryVectorStore() if memory else ChromaVectorStore(persist_dir)
    sym_db = root / ".khaslana" / "symbols.db"
    sym_table = SymbolTable(":memory:") if memory else SymbolTable(sym_db)
    embedder = get_embedding_provider(config)
    chunker = CodeChunker(
        target_chunk_tokens=config.retrieval.chunk_size,
        overlap_tokens=config.retrieval.chunk_overlap,
    )

    updater = IncrementalUpdater(
        store=store,
        embedder=embedder,
        chunker=chunker,
        symbol_table=sym_table,
        root_dir=root,
    )

    if not daemon_mode:
        console.print(f"[bold cyan]🔍 Khaslana live file watcher active on:[/bold cyan] {root}")
        console.print(
            f"[dim]Debounce: {debounce}s. Monitoring files for modifications. "
            "Press Ctrl+C to stop.[/dim]\n"
        )

        with console.status("[bold green]Checking for modified files...[/bold green]"):
            initial_summary = updater.scan_and_update_all()
            if initial_summary.updated_files or initial_summary.deleted_files:
                console.print(
                    f"[green]✓ Initial sync: updated {len(initial_summary.updated_files)} files, "
                    f"deleted {len(initial_summary.deleted_files)} files in "
                    f"{initial_summary.elapsed_seconds}s[/green]\n"
                )


    def _on_change(summary: IncrementalSummary) -> None:
        if summary.updated_files or summary.deleted_files:
            msg = (
                f"[bold green]↻ Index updated:[/bold green] "
                f"+{len(summary.updated_files)} modified, -{len(summary.deleted_files)} deleted "
                f"in {summary.elapsed_seconds}s"
            )
            if not daemon_mode:
                console.print(msg)
                for f in summary.updated_files[:3]:
                    console.print(f"  [dim]• re-indexed: {f}[/dim]")
                if len(summary.updated_files) > 3:
                    console.print(f"  [dim]• ... and {len(summary.updated_files) - 3} more[/dim]")
                for f in summary.deleted_files:
                    console.print(f"  [dim]• removed: {f}[/dim]")

    watcher = RepoWatcher(
        updater=updater,
        debounce_seconds=debounce,
        on_change_callback=_on_change,
    )

    watcher.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        if not daemon_mode:
            console.print("\n[yellow]Stopping file watcher...[/yellow]")
    finally:
        watcher.stop()
        if not daemon_mode:
            console.print("[green]✓ File watcher stopped.[/green]")
