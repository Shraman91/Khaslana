"""Khaslana CLI main entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from khaslana.cli.config_cmd import config_app
from khaslana.cli.doctor import run_doctor
from khaslana.core.config import load_config
from khaslana.core.errors import KhaslanaError
from khaslana.core.llm import ChatMessage, LLMRouter
from khaslana.core.logging import setup_logging
from khaslana.core.prompts import EXPLAIN_PROMPT_TEMPLATE, SYSTEM_PROMPT

# Retrieval imports
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import get_embedding_provider
from khaslana.retrieval.indexer import Indexer
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import ChromaVectorStore, MemoryVectorStore

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


@app.command("index")
def index_cmd(
    path: Annotated[Path, typer.Argument(help="Directory path to index")] = Path("."),
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
    memory: Annotated[
        bool, typer.Option("--memory", help="Use in-memory vector store (ephemeral)")
    ] = False,
) -> None:
    """Index a directory into the local vector database for RAG."""
    config = load_config(cli_overrides={"debug": True} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    console.print(f"[bold blue]Indexing directory:[/bold blue] {path.resolve()}")

    try:
        persist_dir = path.resolve() / config.retrieval.store_dir
        store: Any = MemoryVectorStore() if memory else ChromaVectorStore(persist_dir)

        # Allow fallback to fast hash if real transformers are missing/slow
        embedder = get_embedding_provider(config)
        chunker = CodeChunker(
            target_chunk_tokens=config.retrieval.chunk_size,
            overlap_tokens=config.retrieval.chunk_overlap,
        )
        indexer = Indexer(store, embedder, chunker)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning and indexing files...", total=None)

            def _progress_cb(rel_path: str, curr: int, total: int) -> None:
                progress.update(task, description=f"Indexing ({curr}/{total}): {rel_path}")

            summary = indexer.index_directory(path, progress_callback=_progress_cb)
            progress.update(task, description="[bold green]Indexing complete![/bold green]")

        console.print(
            f"\n[green]✓ Indexed {summary.indexed_files} files "
            f"({summary.total_chunks} chunks)[/green] in {summary.elapsed_seconds}s. "
            f"Skipped {summary.skipped_files} files."
        )

    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(1) from e


@app.command("ask")
def ask(
    query: Annotated[str, typer.Argument(help="Question or prompt to ask Khaslana")],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override LLM model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Override LLM provider")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
    skip_rag: Annotated[bool, typer.Option("--no-rag", help="Skip retrieval augmentation")] = False,
) -> None:
    """Ask Khaslana a technical question with codebase context."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    cwd = Path.cwd()
    persist_dir = cwd / config.retrieval.store_dir

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=query),
    ]

    with console.status("[bold green]Preparing context...[/bold green]"):
        try:
            if not skip_rag and persist_dir.is_dir():
                store = ChromaVectorStore(persist_dir)
                embedder = get_embedding_provider(config)
                pipeline = RetrievalPipeline(store, embedder)

                results = pipeline.search(query, top_k=config.retrieval.top_k)
                if results:
                    messages = pipeline.assemble_context_prompt(query, results, SYSTEM_PROMPT)
        except Exception as e:
            if config.debug:
                console.print(f"[yellow]RAG warning:[/yellow] {e}")

    router = LLMRouter(config)

    async def _run() -> None:
        try:
            provider_info = f"{config.llm.provider}/{config.llm.model}"
            console.print(f"[bold blue]Khaslana ({provider_info}):[/bold blue]")
            async for chunk in router.stream(messages):
                console.print(chunk, end="")
            console.print()
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1) from e

    asyncio.run(_run())


@app.command("explain")
def explain(
    file_path: Annotated[Path, typer.Argument(help="Path to the file to explain")],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Override provider")
    ] = None,
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
        raise typer.Exit(code=1) from e

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
            provider_info = f"{config.llm.provider}/{config.llm.model}"
            console.print(f"[bold blue]Explaining {file_path.name} ({provider_info}):[/bold blue]")
            async for chunk in router.stream(messages):
                console.print(chunk, end="")
            console.print()
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1) from e

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
