"""Factory for setting up agents and wiring core tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from khaslana.core.config import KhaslanaConfig
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.symbols import SymbolTable
from khaslana.retrieval.bm25 import BM25Index
from khaslana.retrieval.embeddings import get_embedding_provider
from khaslana.retrieval.pipeline import RetrievalPipeline
from khaslana.retrieval.store import ChromaVectorStore
from khaslana.tools.base import ToolRegistry
from khaslana.tools.core import (
    FindSymbolTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    SearchCodeTool,
    WhereUsedTool,
)


def create_agent_tool_registry(
    root_dir: Path | str = ".",
    config: KhaslanaConfig | None = None,
    confirmation_handler: Callable[[str], bool] | None = None,
) -> ToolRegistry:
    """Instantiate and register all core tools for the project workspace."""
    root = Path(root_dir).resolve()
    registry = ToolRegistry()

    # 1. Filesystem & Execution
    registry.register(ReadFileTool(root_dir=root))
    registry.register(ListFilesTool(root_dir=root))
    registry.register(
        RunCommandTool(
            root_dir=root,
            confirmation_handler=confirmation_handler,
        )
    )

    # 2. Code Intelligence (Symbol table & Call graph)
    sym_db_path = root / ".khaslana" / "symbols.db"
    if sym_db_path.is_file():
        sym_table = SymbolTable(sym_db_path)
        call_graph = CallGraph(sym_table)
        registry.register(FindSymbolTool(sym_table))
        registry.register(WhereUsedTool(sym_table, call_graph))
    else:
        # Ephemeral empty symbol table so tools are always present
        sym_table = SymbolTable(":memory:")
        call_graph = CallGraph(sym_table)
        registry.register(FindSymbolTool(sym_table))
        registry.register(WhereUsedTool(sym_table, call_graph))

    # 3. Hybrid Retrieval
    pipeline: RetrievalPipeline | None = None
    if config:
        persist_dir = root / config.retrieval.store_dir
        if persist_dir.is_dir():
            try:
                store = ChromaVectorStore(persist_dir)
                embedder = get_embedding_provider(config)
                # Build or load BM25 index from chunks in store
                all_chunks = store.get_all_chunks()
                bm25 = BM25Index(all_chunks) if all_chunks else None
                pipeline = RetrievalPipeline(
                    vector_store=store,
                    embedding_provider=embedder,
                    bm25_index=bm25,
                )
            except Exception:
                pipeline = None

    registry.register(SearchCodeTool(pipeline=pipeline, root_dir=root))
    return registry
