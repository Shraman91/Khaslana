"""Unit tests for file watching, incremental indexing, and daemon management (Phase 8)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from typer.testing import CliRunner
from watchdog.events import FileModifiedEvent

from khaslana.cli.main import app
from khaslana.intelligence.symbols import SymbolTable
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import FastHashEmbedding
from khaslana.retrieval.models import FileRecord, IndexState, compute_file_hash
from khaslana.retrieval.store import MemoryVectorStore
from khaslana.retrieval.watcher import (
    DaemonManager,
    DebouncedEventHandler,
    IncrementalUpdater,
    RepoWatcher,
    is_pid_alive,
)

runner = CliRunner()


def test_file_hash_and_state(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    h1 = compute_file_hash(f)
    assert len(h1) == 64

    # Change content -> new hash
    f.write_text("def hello():\n    return 'khaslana'\n", encoding="utf-8")
    h2 = compute_file_hash(f)
    assert h1 != h2

    # Test IndexState load & save
    state_file = tmp_path / "state.json"
    state = IndexState(root_dir=str(tmp_path))
    state.files["sample.py"] = FileRecord(hash=h2, mtime=123.45, size=40)
    state.save(state_file)

    loaded = IndexState.load(state_file)
    assert loaded is not None
    assert loaded.root_dir == str(tmp_path)
    assert "sample.py" in loaded.files
    assert loaded.files["sample.py"].hash == h2


def test_incremental_updater_lifecycle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    file_a = repo / "module_a.py"
    file_a.write_text("def alpha():\n    return 1\n", encoding="utf-8")

    file_b = repo / "module_b.py"
    file_b.write_text("def beta():\n    return 2\n", encoding="utf-8")

    store = MemoryVectorStore()
    embedder = FastHashEmbedding(dim=16)
    sym_table = SymbolTable(":memory:")
    chunker = CodeChunker()

    updater = IncrementalUpdater(
        store=store,
        embedder=embedder,
        chunker=chunker,
        symbol_table=sym_table,
        root_dir=repo,
        state_file=repo / "state.json",
    )

    # 1. Initial scan
    summary = updater.scan_and_update_all()
    assert len(summary.updated_files) == 2
    assert "module_a.py" in summary.updated_files
    assert "module_b.py" in summary.updated_files
    assert store.count() >= 2
    assert sym_table.count_symbols() >= 2
    assert not updater.is_stale()

    # 2. No changes -> nothing updated
    summary_noop = updater.scan_and_update_all()
    assert len(summary_noop.updated_files) == 0
    assert summary_noop.unchanged_files == 2
    assert not updater.is_stale()

    # 3. Modify module_a.py
    file_a.write_text("def alpha():\n    # modified\n    return 42\n", encoding="utf-8")
    assert updater.is_stale()

    mod_summary = updater.scan_and_update_all()
    assert len(mod_summary.updated_files) == 1
    assert "module_a.py" in mod_summary.updated_files
    assert not updater.is_stale()

    # 4. Delete module_b.py
    file_b.unlink()
    assert updater.is_stale()

    del_summary = updater.scan_and_update_all()
    assert len(del_summary.deleted_files) == 1
    assert "module_b.py" in del_summary.deleted_files
    assert not updater.is_stale()
    assert len(store.get_all_chunks()) >= 1
    # Verify module_b symbols deleted
    assert len(sym_table.find_definitions("beta")) == 0


def test_debounced_event_handler(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    file_x = repo / "x.py"
    file_x.write_text("x = 10\n", encoding="utf-8")

    store = MemoryVectorStore()
    embedder = FastHashEmbedding(dim=16)
    sym_table = SymbolTable(":memory:")

    updater = IncrementalUpdater(
        store=store,
        embedder=embedder,
        symbol_table=sym_table,
        root_dir=repo,
        state_file=repo / "state.json",
    )
    updater.scan_and_update_all()

    updates: list[list[str]] = []

    def on_batch(summary: Any) -> None:
        updates.append(summary.updated_files)

    handler = DebouncedEventHandler(
        updater=updater,
        debounce_seconds=0.05,
        on_change_callback=on_batch,
    )

    file_x.write_text("x = 99\n", encoding="utf-8")
    event = FileModifiedEvent(str(file_x))
    handler.on_modified(event)
    handler.on_modified(event)
    handler.on_modified(event)

    # Wait for debounce timer to fire
    for _ in range(30):
        if updates:
            break
        time.sleep(0.05)

    assert len(updates) == 1
    assert "x.py" in updates[0]





def test_repo_watcher_start_stop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    store = MemoryVectorStore()
    embedder = FastHashEmbedding(dim=16)
    updater = IncrementalUpdater(
        store=store,
        embedder=embedder,
        root_dir=repo,
        state_file=repo / "state.json",
    )

    watcher = RepoWatcher(updater=updater, debounce_seconds=0.1)
    assert not watcher.is_alive()

    watcher.start()
    assert watcher.is_alive()

    watcher.stop()
    assert not watcher.is_alive()


def test_daemon_manager_lifecycle(tmp_path: Path) -> None:
    manager = DaemonManager(root_dir=tmp_path)

    # Initially stopped
    st = manager.status()
    assert not st["running"]
    assert st["pid"] is None

    # Write a fake dead PID
    fake_pid = 9999999
    manager.pid_file.parent.mkdir(parents=True, exist_ok=True)
    manager.pid_file.write_text(f'{{"pid": {fake_pid}, "started_at": 1000.0}}', encoding="utf-8")

    # Status should notice process is dead and clean up stale PID file
    st_stale = manager.status()
    assert not st_stale["running"]
    assert st_stale.get("stale_cleaned") is True
    assert not manager.pid_file.exists()


def test_is_pid_alive_nonexistent() -> None:
    assert not is_pid_alive(-1)
    assert not is_pid_alive(99999999)


def test_cli_daemon_commands() -> None:
    # Test daemon status command
    res_status = runner.invoke(app, ["daemon", "status"])
    assert res_status.exit_code == 0
    assert "STOPPED" in res_status.output or "Daemon Status" in res_status.output

    # Test daemon stop when not running
    res_stop = runner.invoke(app, ["daemon", "stop"])
    assert res_stop.exit_code == 0
    assert "not" in res_stop.output.lower() or "no" in res_stop.output.lower()
