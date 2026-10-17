"""Khaslana CLI main entry point."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from khaslana.agents.factory import create_agent_tool_registry
from khaslana.agents.loop import AgentLoop
from khaslana.cli.config_cmd import config_app
from khaslana.cli.doctor import run_doctor
from khaslana.core.config import load_config
from khaslana.core.errors import KhaslanaError
from khaslana.core.llm import ChatMessage, LLMRouter
from khaslana.core.logging import setup_logging
from khaslana.core.prompts import EXPLAIN_PROMPT_TEMPLATE, SYSTEM_PROMPT
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.symbols import SymbolTable

# Retrieval imports
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import get_embedding_provider
from khaslana.retrieval.indexer import Indexer
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import ChromaVectorStore, MemoryVectorStore
from khaslana.sandbox.docker_runner import DockerRunner
from khaslana.sandbox.fix_loop import SelfCorrectingLoop
from khaslana.tools.review import (
    CodeReviewer,
    format_review_json,
    format_review_markdown,
    format_review_rich,
)

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
    """Index a directory into the local vector database for RAG and code intelligence."""
    config = load_config(cli_overrides={"debug": True} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    console.print(f"[bold blue]Indexing directory:[/bold blue] {path.resolve()}")

    try:
        persist_dir = path.resolve() / config.retrieval.store_dir
        store: Any = MemoryVectorStore() if memory else ChromaVectorStore(persist_dir)

        sym_db_path = path.resolve() / ".khaslana" / "symbols.db"
        sym_table = SymbolTable(":memory:") if memory else SymbolTable(sym_db_path)

        embedder = get_embedding_provider(config)
        chunker = CodeChunker(
            target_chunk_tokens=config.retrieval.chunk_size,
            overlap_tokens=config.retrieval.chunk_overlap,
        )
        indexer = Indexer(
            store,
            embedder,
            chunker,
            symbol_table=sym_table,
        )

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
            f"({summary.total_chunks} chunks, {sym_table.count_symbols()} symbols, "
            f"{sym_table.count_calls()} call edges)[/green] in {summary.elapsed_seconds}s. "
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
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            "-d",
            help="Deep analysis combining symbols, call graph neighbors, and retrieved context",
        ),
    ] = False,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Override provider")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show debug logs")] = False,
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

    if deep:
        sym_db_path = Path.cwd() / ".khaslana" / "symbols.db"
        if sym_db_path.is_file():
            sym_table = SymbolTable(sym_db_path)
            call_graph = CallGraph(sym_table)
            symbols = sym_table.get_symbols_in_file(str(file_path).replace("\\", "/"))

            deep_intel: list[str] = [
                "\n--- Deep Code Intelligence ---",
                f"Symbols defined in this file ({len(symbols)}):",
            ]
            for s in symbols:
                callers = call_graph.get_callers(s.name)
                callees = call_graph.get_callees(s.name)
                callers_str = f" | Callers: {', '.join(callers)}" if callers else ""
                callees_str = f" | Calls: {', '.join(callees)}" if callees else ""
                deep_intel.append(
                    f"- [{s.symbol_type}] {s.name} (lines {s.line_range[0]}-{s.line_range[1]})"
                    f"{callers_str}{callees_str}"
                )

            prompt += "\n" + "\n".join(deep_intel)

    router = LLMRouter(config)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt),
    ]

    async def _run() -> None:
        try:
            provider_info = f"{config.llm.provider}/{config.llm.model}"
            analysis_type = "Deeply explaining" if deep else "Explaining"
            console.print(
                f"[bold blue]{analysis_type} {file_path.name} ({provider_info}):[/bold blue]"
            )
            async for chunk in router.stream(messages):
                console.print(chunk, end="")
            console.print()
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1) from e

    asyncio.run(_run())


@app.command("where-used")
def where_used_cmd(
    symbol_name: Annotated[str, typer.Argument(help="Symbol identifier to search for")],
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Find all definitions and call site references for a symbol."""
    config = load_config(cli_overrides={"debug": True} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_dir = path or Path.cwd()
    sym_db_path = target_dir / ".khaslana" / "symbols.db"
    if not sym_db_path.is_file():
        console.print(
            "[bold yellow]Warning:[/bold yellow] No symbol database found. "
            "Run [bold green]khaslana index[/bold green] first."
        )
        raise typer.Exit(code=1)

    try:
        sym_table = SymbolTable(sym_db_path)
        defs = sym_table.find_definitions(symbol_name)
        refs = sym_table.find_references(symbol_name)

        console.print(f"\n[bold blue]Symbol Lookup:[/bold blue] [cyan]{symbol_name}[/cyan]\n")

        # 1. Definitions
        if defs:
            console.print(f"[bold green]Definitions ({len(defs)}):[/bold green]")
            for d in defs:
                parent = f"{d.parent_symbol}." if d.parent_symbol else ""
                sig = f"{d.signature}" if d.signature else ""
                loc = f"{d.file}:{d.line_range[0]}-{d.line_range[1]}"
                sym_type = f"[bold cyan]{d.symbol_type}[/bold cyan]"
                sym_name = f"[bold]{parent}{d.name}{sig}[/bold]"
                console.print(f"  • {sym_type} {sym_name} at [dim green]{loc}[/dim green]")
                if d.docstring:
                    console.print(f"    [dim italic]{d.docstring.strip()}[/dim italic]")
        else:
            console.print("[dim]No definition found in indexed symbols.[/dim]")

        # 2. References / Call Sites
        console.print()
        if refs:
            console.print(f"[bold green]References & Call Sites ({len(refs)}):[/bold green]")
            # Group by file
            by_file: dict[str, list[tuple[str, int]]] = {}
            for r in refs:
                by_file.setdefault(r.caller_file, []).append((r.caller_symbol, r.line_number))

            for file_path, call_sites in by_file.items():
                console.print(f"  [bold underline]{file_path}[/bold underline]:")
                for caller_sym, line_no in call_sites:
                    console.print(f"    • Line {line_no} in [magenta]{caller_sym}[/magenta]()")
        else:
            console.print("[dim]No call site references found.[/dim]")

    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(code=1) from e


@app.command("graph")
def graph_cmd(
    symbol_name: Annotated[str, typer.Argument(help="Root symbol to trace")],
    depth: Annotated[int, typer.Option("--depth", "-d", help="Call traversal depth")] = 2,
    format_type: Annotated[
        str, typer.Option("--format", "-f", help="Output format: mermaid, dot, or text")
    ] = "mermaid",
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show debug logs")] = False,
) -> None:
    """Generate caller/callee call graph around a symbol."""
    config = load_config(cli_overrides={"debug": True} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_dir = path or Path.cwd()
    sym_db_path = target_dir / ".khaslana" / "symbols.db"
    if not sym_db_path.is_file():
        console.print(
            "[bold yellow]Warning:[/bold yellow] No symbol database found. "
            "Run [bold green]khaslana index[/bold green] first."
        )
        raise typer.Exit(code=1)

    try:
        sym_table = SymbolTable(sym_db_path)
        call_graph = CallGraph(sym_table)

        fmt = format_type.lower()
        if fmt == "dot":
            dot_out = call_graph.export_dot(symbol_name, depth=depth)
            console.print(dot_out)
        elif fmt == "text":
            callers = call_graph.get_callers(symbol_name, max_depth=depth)
            callees = call_graph.get_callees(symbol_name, max_depth=depth)
            console.print(f"[bold cyan]Call Graph for {symbol_name} (depth={depth}):[/bold cyan]")
            console.print(f"  [bold]Incoming Callers:[/bold] {', '.join(callers) or 'None'}")
            console.print(f"  [bold]Outgoing Callees:[/bold] {', '.join(callees) or 'None'}")
        else:
            # Default mermaid
            mermaid_out = call_graph.export_mermaid(symbol_name, depth=depth)
            console.print(mermaid_out)

    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(code=1) from e


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Code search query or keywords")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of results")] = 5,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain",
            "-e",
            help="Show detailed scoring breakdown (semantic, BM25, boost)",
        ),
    ] = False,
    file: Annotated[
        str | None, typer.Option("--file", "-f", help="Prioritize matches in specific file")
    ] = None,
    fusion: Annotated[
        str, typer.Option("--fusion", help="Fusion strategy: weighted or rrf")
    ] = "weighted",
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Perform hybrid code search over indexed repository."""
    config = load_config(cli_overrides={"debug": True} if debug else None)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    cwd = Path.cwd()
    persist_dir = cwd / config.retrieval.store_dir

    if not persist_dir.is_dir():
        console.print(
            "[bold yellow]Warning:[/bold yellow] No index found at "
            f"[cyan]{persist_dir}[/cyan]. Run [bold green]khaslana index[/bold green] first."
        )
        raise typer.Exit(code=1)

    try:
        store = ChromaVectorStore(persist_dir)
        embedder = get_embedding_provider(config)
        pipeline = RetrievalPipeline(store, embedder)

        fusion_mode: Literal["weighted", "rrf"] = (
            "rrf" if fusion.lower() == "rrf" else "weighted"
        )
        results = pipeline.search(
            query=query,
            top_k=top_k,
            fusion=fusion_mode,
            focused_file=file,
        )

        if not results:
            console.print(f"[yellow]No matching code found for query:[/yellow] {query}")
            return

        console.print(
            f"\n[bold green]Found {len(results)} matches for:[/bold green] \"{query}\"\n"
        )

        if explain:
            table = Table(title="Search Results & Score Breakdown", show_header=True)
            table.add_column("Rank", justify="right", style="cyan", width=4)
            table.add_column("File:Lines", style="bold green", width=30)
            table.add_column("Symbol", style="magenta", width=18)
            table.add_column("Semantic", justify="right", width=10)
            table.add_column("BM25", justify="right", width=10)
            table.add_column("Boost", justify="right", width=8)
            table.add_column("Final Score", justify="right", style="bold yellow", width=12)

            for rank, r in enumerate(results, 1):
                chunk = r.chunk
                file_lines = f"{chunk.file}:{chunk.line_range[0]}-{chunk.line_range[1]}"
                sem_str = f"{r.semantic_score:.4f}" if r.semantic_score is not None else "N/A"
                bm25_str = f"{r.bm25_score:.4f}" if r.bm25_score is not None else "N/A"
                table.add_row(
                    str(rank),
                    file_lines,
                    chunk.symbol or "-",
                    sem_str,
                    bm25_str,
                    f"{r.boost:.2f}x",
                    f"{r.score:.4f}",
                )

            console.print(table)
        else:
            for i, r in enumerate(results, 1):
                chunk = r.chunk
                sym_info = f" ({chunk.symbol})" if chunk.symbol else ""
                loc = f"{chunk.file}:{chunk.line_range[0]}-{chunk.line_range[1]}"
                header = (
                    f"[bold cyan]{i}.[/bold cyan] [bold green]{loc}[/bold green]"
                    f"{sym_info} [dim](score: {r.score:.4f})[/dim]"
                )
                console.print(header)
                preview = "\n".join(chunk.text.splitlines()[:5])
                console.print(Panel(preview, border_style="dim"))

    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(code=1) from e


@app.command("debug")
def debug_cmd(
    log_file: Annotated[Path, typer.Argument(help="Path to error log or traceback file")],
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root directory")
    ] = None,
    max_iters: Annotated[
        int, typer.Option("--max-iters", "-i", help="Maximum agent iterations")
    ] = 10,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override LLM model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override LLM provider")
    ] = None,
    auto_approve: Annotated[
        bool, typer.Option("--yes", "-y", help="Auto-approve mutating commands")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Analyze an error log, locate relevant code, hypothesize root causes, and propose fixes."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_root = (path or Path.cwd()).resolve()

    if not log_file.exists():
        console.print(f"[bold red]Error:[/bold red] Error log file '{log_file}' does not exist.")
        raise typer.Exit(code=1)

    try:
        log_content = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(code=1) from e

    def _confirm(cmd: str) -> bool:
        if auto_approve:
            return True
        return typer.confirm(f"Agent wants to execute: '{cmd}'. Allow?", default=False)

    router = LLMRouter(config)
    tools = create_agent_tool_registry(
        root_dir=target_root,
        config=config,
        confirmation_handler=_confirm,
    )

    loop = AgentLoop(
        llm=router,
        tool_registry=tools,
        max_iterations=max_iters,
        console=console,
        verbose=True,
    )

    task = (
        "Analyze the following error traceback and resolve the issue:\n\n"
        f"```\n{log_content}\n```\n\n"
        "Instructions:\n"
        "1. Identify the failing files, functions, and lines.\n"
        "2. Inspect the relevant source files with read_file or search_code.\n"
        "3. Determine the root cause of the bug.\n"
        "4. Propose a concrete fix with explanation and code diff/patch."
    )

    console.print(f"[bold blue]Starting agent debugger on:[/bold blue] {log_file.name}\n")
    try:
        asyncio.run(loop.run(task))
    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(code=1) from e


@app.command("analyze")
def analyze_cmd(
    target: Annotated[
        str, typer.Argument(help="Target to analyze (default: 'project')")
    ] = "project",
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root directory")
    ] = None,
    max_iters: Annotated[
        int, typer.Option("--max-iters", "-i", help="Maximum agent iterations")
    ] = 10,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override LLM model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override LLM provider")
    ] = None,
    auto_approve: Annotated[
        bool, typer.Option("--yes", "-y", help="Auto-approve mutating commands")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Perform architectural analysis, inspect entry points, and highlight risk areas."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_root = (path or Path.cwd()).resolve()

    def _confirm(cmd: str) -> bool:
        if auto_approve:
            return True
        return typer.confirm(f"Agent wants to execute: '{cmd}'. Allow?", default=False)

    router = LLMRouter(config)
    tools = create_agent_tool_registry(
        root_dir=target_root,
        config=config,
        confirmation_handler=_confirm,
    )

    loop = AgentLoop(
        llm=router,
        tool_registry=tools,
        max_iterations=max_iters,
        console=console,
        verbose=True,
    )

    task = (
        f"Perform an architectural analysis of this {target}:\n"
        "1. List files and discover entry points, modules, and structure.\n"
        "2. Examine package/project configurations and dependency overview.\n"
        "3. Summarize architecture, design patterns, and module boundaries.\n"
        "4. Highlight potential risk areas, security concerns, or architectural debt."
    )

    console.print(f"[bold blue]Starting architectural analysis on:[/bold blue] {target_root}\n")
    try:
        asyncio.run(loop.run(task))
    except Exception as e:
        _handle_error(e, config.debug)
        raise typer.Exit(code=1) from e


@app.command("review")
def review_cmd(
    staged: Annotated[
        bool, typer.Option("--staged", help="Review staged changes")
    ] = False,
    branch: Annotated[
        str | None, typer.Option("--branch", "-b", help="Review changes against branch")
    ] = None,
    format_type: Annotated[
        str, typer.Option("--format", "-f", help="Output format: rich, json, or markdown")
    ] = "rich",
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root directory")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override LLM model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override LLM provider")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Perform a diff-aware code review reporting bugs, security, perf, and style findings."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_root = (path or Path.cwd()).resolve()
    router = LLMRouter(config)
    reviewer = CodeReviewer(router=router, root_dir=target_root, console=console)

    async def _run() -> None:
        try:
            report = await reviewer.review(staged=staged, branch=branch)
            fmt = format_type.lower()
            if fmt == "json":
                console.print(format_review_json(report))
            elif fmt == "markdown":
                console.print(format_review_markdown(report))
            else:
                format_review_rich(report, console)
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1) from e

    asyncio.run(_run())


@app.command("explain-diff")
def explain_diff_cmd(
    staged: Annotated[
        bool, typer.Option("--staged", help="Explain staged changes")
    ] = False,
    branch: Annotated[
        str | None, typer.Option("--branch", "-b", help="Explain changes against branch")
    ] = None,
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root directory")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override LLM model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override LLM provider")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Explain the changes introduced by a git diff and their system impact."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_root = (path or Path.cwd()).resolve()
    router = LLMRouter(config)
    reviewer = CodeReviewer(router=router, root_dir=target_root, console=console)

    async def _run() -> None:
        try:
            explanation = await reviewer.explain_diff(staged=staged, branch=branch)
            console.print(
                Panel(
                    explanation,
                    title="[bold blue]Git Diff Explanation[/bold blue]",
                    border_style="blue",
                )
            )
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1) from e

    asyncio.run(_run())


@app.command("commit-msg")
def commit_msg_cmd(
    path: Annotated[
        Path | None, typer.Option("--path", "-p", help="Target project root directory")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override LLM model name")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override LLM provider")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Show debug logs")] = False,
) -> None:
    """Generate a Conventional Commit message from staged git changes."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_root = (path or Path.cwd()).resolve()
    router = LLMRouter(config)
    reviewer = CodeReviewer(router=router, root_dir=target_root, console=console)

    async def _run() -> None:
        try:
            msg = await reviewer.generate_commit_message()
            if not msg:
                console.print(
                    "[yellow]No staged changes found. Run `git add <file>` first.[/yellow]"
                )
                raise typer.Exit(code=1)
            console.print(msg)
        except Exception as e:
            _handle_error(e, config.debug)
            raise typer.Exit(code=1) from e

    asyncio.run(_run())


@app.command("fix")
def fix_code(
    file_path: Annotated[Path, typer.Argument(help="Path to the file to fix")],
    test_cmd: Annotated[
        str | None,
        typer.Option("--test-cmd", "-t", help="Command to run tests (auto-detected if omitted)"),
    ] = None,
    max_attempts: Annotated[
        int,
        typer.Option("--max-attempts", "-m", help="Maximum fix attempts in sandbox"),
    ] = 3,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Apply fix automatically without confirmation prompt"),
    ] = False,
    model: Annotated[str | None, typer.Option("--model", help="Model to use")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Provider to use")] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Show debug logs and tracebacks"),
    ] = False,
) -> None:
    """Diagnose and fix code issues using isolated ephemeral sandbox containers."""
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if provider:
        overrides["provider"] = provider
    if debug:
        overrides["debug"] = True

    config = load_config(cli_overrides=overrides)
    setup_logging(level="DEBUG" if config.debug else config.log_level, debug=config.debug)

    target_file = file_path.resolve()
    cwd_root = Path.cwd().resolve()

    try:
        rel_path = target_file.relative_to(cwd_root)
        project_root = cwd_root
    except ValueError:
        project_root = target_file.parent
        rel_path = Path(target_file.name)

    router = LLMRouter(config)
    runner = DockerRunner()

    async def _run() -> None:
        try:
            console.print(
                f"[bold blue]Starting sandbox fix loop for "
                f"[cyan]{rel_path.as_posix()}[/cyan]...[/bold blue]"
            )
            loop = SelfCorrectingLoop(
                runner=runner,
                llm=router,
                project_root=project_root,
                file_rel_path=rel_path,
                test_cmd=test_cmd,
                max_attempts=max_attempts,
            )
            session = await loop.run()

            if not session.attempts:
                if session.success:
                    console.print(
                        "[bold green]✓ Tests already pass! No fix required.[/bold green]"
                    )
                    return
                console.print("[yellow]No attempts could be made.[/yellow]")
                return

            for attempt in session.attempts:
                status = (
                    "[bold green]PASS[/bold green]"
                    if attempt.success
                    else "[bold red]FAIL[/bold red]"
                )
                syntax_diff = Syntax(
                    attempt.diff or "(No diff)", "diff", theme="monokai", line_numbers=True
                )
                panel_title = (
                    f"Attempt {attempt.attempt_number}/{max_attempts} - Test Status: {status}"
                )
                console.print(
                    Panel(
                        syntax_diff,
                        title=panel_title,
                        border_style="green" if attempt.success else "red",
                    )
                )

            if session.success and session.final_code:
                console.print(
                    "[bold green]✓ A passing fix was verified inside the sandbox![/bold green]"
                )
                apply_fix = yes
                if not apply_fix:
                    apply_fix = typer.confirm(
                        f"Apply this fix to {rel_path.as_posix()}?", default=False
                    )

                if apply_fix:
                    target_file.write_text(session.final_code, encoding="utf-8")
                    console.print(
                        f"[bold green]✓ Fix applied to {rel_path.as_posix()}[/bold green]"
                    )
                else:
                    console.print("[yellow]Fix discarded. Working tree untouched.[/yellow]")
            else:
                console.print(
                    f"[bold red]✗ Could not fix {rel_path.as_posix()} "
                    f"after {max_attempts} attempts. Working tree untouched.[/bold red]"
                )
                raise typer.Exit(code=1)

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
