from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from khaslana.core.errors import DaemonError
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.parser import parse_source
from khaslana.intelligence.symbols import SymbolTable
from khaslana.retrieval.chunker import CodeChunker
from khaslana.retrieval.embeddings import EmbeddingProvider
from khaslana.retrieval.indexer import (
    EXTENSION_TO_LANGUAGE,
    is_binary_file,
    load_gitignore_spec,
)
from khaslana.retrieval.models import (
    INDEX_STATE_FILE,
    FileRecord,
    IncrementalSummary,
    IndexState,
    compute_file_hash,
)
from khaslana.retrieval.store import VectorStore

DAEMON_PID_FILE = ".khaslana/daemon.pid"
DAEMON_LOG_FILE = ".khaslana/daemon.log"




class IncrementalUpdater:
    """Performs hash-checked incremental updates to vector store and symbol table."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        chunker: CodeChunker | None = None,
        symbol_table: SymbolTable | None = None,
        root_dir: Path | str = ".",
        state_file: Path | None = None,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or CodeChunker()
        self.symbol_table = symbol_table
        self.root_dir = Path(root_dir).resolve()
        self.state_file = state_file or (self.root_dir / INDEX_STATE_FILE)
        self.max_file_size_bytes = max_file_size_bytes
        self.spec = load_gitignore_spec(self.root_dir)

        # Load existing state or initialize new
        loaded = IndexState.load(self.state_file)
        if loaded and Path(loaded.root_dir).resolve() == self.root_dir:
            self.state = loaded
        else:
            self.state = IndexState(root_dir=str(self.root_dir))

    def should_ignore(self, rel_path: str, is_dir: bool = False) -> bool:
        """Check if relative path matches gitignore or default ignore patterns."""
        try:
            full = (self.root_dir / rel_path).resolve()
            if full == self.state_file.resolve():
                return True
        except Exception:
            pass

        parts = rel_path.split("/")
        ignored_names = {
            ".git",
            "node_modules",
            "venv",
            ".venv",
            "__pycache__",
            ".khaslana",
            "dist",
            "build",
            ".idea",
            ".vscode",
        }
        if any(part in ignored_names for part in parts):
            return True

        if is_dir:
            return self.spec.match_file(f"{rel_path}/")
        return self.spec.match_file(rel_path)


    def update_file(self, full_path: Path) -> bool:
        """Incrementally index or re-index a single file if its hash has changed."""
        if not full_path.is_file():
            return False

        try:
            rel_path = full_path.resolve().relative_to(self.root_dir).as_posix()
        except ValueError:
            return False

        if self.should_ignore(rel_path):
            return False

        # File size validation
        try:
            stat = full_path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
            if size > self.max_file_size_bytes or size == 0:
                self.delete_file(rel_path)
                return False
        except Exception:
            return False

        # Binary check
        if is_binary_file(full_path):
            self.delete_file(rel_path)
            return False

        # Compute hash
        try:
            new_hash = compute_file_hash(full_path)
        except Exception:
            return False

        # Check if unchanged
        existing_record = self.state.files.get(rel_path)
        if existing_record and existing_record.hash == new_hash:
            return False

        # 1. Delete previous chunks from store
        self.store.delete_file(rel_path)

        # 2. Read content & re-chunk
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        ext = full_path.suffix.lower()
        lang = EXTENSION_TO_LANGUAGE.get(ext, "plaintext")

        # 3. Update symbol table
        if self.symbol_table:
            try:
                parsed = parse_source(rel_path, content, language=lang)
                self.symbol_table.index_parsed_file(parsed)
            except Exception:
                pass

        # 4. Chunk file
        chunks = self.chunker.chunk_file(
            file_path=rel_path,
            content=content,
            language=lang,
            mtime=mtime,
        )

        # 5. Enrich chunks with call graph if symbol table is available
        if self.symbol_table and chunks:
            call_graph = CallGraph(self.symbol_table)
            for chunk in chunks:
                if chunk.symbol and chunk.symbol != "<module_scope>":
                    chunk.callers = call_graph.get_callers(chunk.symbol, max_depth=1)
                    chunk.callees = call_graph.get_callees(chunk.symbol, max_depth=1)

        # 6. Embed and store new chunks
        if chunks:
            texts = [c.text for c in chunks]
            embeddings = self.embedder.encode(texts)
            self.store.add_chunks(chunks, embeddings)

        # 7. Update state
        self.state.files[rel_path] = FileRecord(hash=new_hash, mtime=mtime, size=size)
        self.state.last_indexed_at = time.time()
        self.state.save(self.state_file)
        return True

    def delete_file(self, rel_path: str) -> bool:
        """Remove file from vector store, symbol table, and state."""
        changed = False
        if rel_path in self.state.files:
            del self.state.files[rel_path]
            changed = True

        self.store.delete_file(rel_path)
        if self.symbol_table:
            self.symbol_table.delete_file(rel_path)

        if changed:
            self.state.last_indexed_at = time.time()
            self.state.save(self.state_file)
        return changed

    def apply_changes(
        self, modified_paths: list[Path], deleted_rel_paths: list[str]
    ) -> IncrementalSummary:
        """Apply a batch of detected file modifications and deletions."""
        start_time = time.time()
        summary = IncrementalSummary(root_dir=str(self.root_dir))

        for full_path in modified_paths:
            with contextlib.suppress(Exception):
                rel = full_path.resolve().relative_to(self.root_dir).as_posix()
                if self.update_file(full_path):
                    summary.updated_files.append(rel)
                else:
                    summary.unchanged_files += 1




        for rel in deleted_rel_paths:
            if self.delete_file(rel):
                summary.deleted_files.append(rel)

        summary.elapsed_seconds = round(time.time() - start_time, 3)
        return summary

    def scan_and_update_all(
        self, progress_callback: Callable[[str, int, int], None] | None = None
    ) -> IncrementalSummary:
        """Scan entire root directory and incrementally update any modified or deleted files."""
        start_time = time.time()
        summary = IncrementalSummary(root_dir=str(self.root_dir))
        current_disk_files: set[str] = set()
        files_to_check: list[Path] = []

        # 1. Discover all disk files
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            rel_dir = Path(dirpath).relative_to(self.root_dir)

            # Prune ignored directories in place
            dirnames[:] = [
                d
                for d in dirnames
                if not self.should_ignore(
                    (rel_dir / d).as_posix() if (rel_dir / d).as_posix() != "." else d, is_dir=True
                )
            ]

            for fname in filenames:
                rel_path = (rel_dir / fname).as_posix() if str(rel_dir) != "." else fname
                if self.should_ignore(rel_path):
                    continue

                full_path = Path(dirpath) / fname
                current_disk_files.add(rel_path)
                files_to_check.append(full_path)

        total_files = len(files_to_check)

        # 2. Update modified or newly added files
        for i, full_path in enumerate(files_to_check):
            rel = full_path.resolve().relative_to(self.root_dir).as_posix()
            if progress_callback:
                progress_callback(rel, i + 1, total_files)

            if self.update_file(full_path):
                summary.updated_files.append(rel)
            else:
                summary.unchanged_files += 1

        # 3. Detect and delete files removed from disk
        stale_records = [
            rel for rel in list(self.state.files.keys()) if rel not in current_disk_files
        ]
        for rel in stale_records:
            if self.delete_file(rel):
                summary.deleted_files.append(rel)

        summary.elapsed_seconds = round(time.time() - start_time, 3)
        return summary

    def is_stale(self) -> bool:
        """Check whether the indexed state differs from the current disk files."""
        if not self.state_file.is_file() or not self.state.files:
            return True

        current_disk_files: set[str] = set()

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            rel_dir = Path(dirpath).relative_to(self.root_dir)
            dirnames[:] = [
                d
                for d in dirnames
                if not self.should_ignore(
                    (rel_dir / d).as_posix() if (rel_dir / d).as_posix() != "." else d, is_dir=True
                )
            ]

            for fname in filenames:
                rel_path = (rel_dir / fname).as_posix() if str(rel_dir) != "." else fname
                if self.should_ignore(rel_path):
                    continue

                current_disk_files.add(rel_path)
                full_path = Path(dirpath) / fname

                record = self.state.files.get(rel_path)
                if not record:
                    return True

                try:
                    stat = full_path.stat()
                    if (
                        stat.st_mtime != record.mtime or stat.st_size != record.size
                    ) and compute_file_hash(full_path) != record.hash:
                        return True
                except Exception:
                    return True

        # Check if any recorded file was deleted
        return any(rel not in current_disk_files for rel in self.state.files)



class DebouncedEventHandler(FileSystemEventHandler):
    """Watches filesystem events with thread-safe debouncing."""

    def __init__(
        self,
        updater: IncrementalUpdater,
        debounce_seconds: float = 0.5,
        on_change_callback: Callable[[IncrementalSummary], None] | None = None,
    ) -> None:
        super().__init__()
        self.updater = updater
        self.debounce_seconds = debounce_seconds
        self.on_change_callback = on_change_callback

        self._pending_modified: set[Path] = set()
        self._pending_deleted: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _normalize_path(self, raw_path: bytes | str) -> Path:
        path_str = (
            raw_path.decode("utf-8", errors="replace")
            if isinstance(raw_path, bytes)
            else raw_path
        )
        return Path(path_str)

    def _should_ignore_path(self, raw_path: bytes | str) -> bool:
        p = self._normalize_path(raw_path)
        try:
            rel = p.resolve().relative_to(self.updater.root_dir).as_posix()
        except ValueError:
            return True
        return self.updater.should_ignore(rel)

    def _schedule_flush(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_seconds, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            modified = list(self._pending_modified)
            deleted = list(self._pending_deleted)
            self._pending_modified.clear()
            self._pending_deleted.clear()
            self._timer = None

        if not modified and not deleted:
            return

        summary = self.updater.apply_changes(modified, deleted)
        if self.on_change_callback and (summary.updated_files or summary.deleted_files):
            with contextlib.suppress(Exception):
                self.on_change_callback(summary)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore_path(event.src_path):
            return
        with self._lock:
            p = self._normalize_path(event.src_path)
            self._pending_modified.add(p)
            try:
                rel = p.resolve().relative_to(self.updater.root_dir).as_posix()
                self._pending_deleted.discard(rel)
            except ValueError:
                pass
            self._schedule_flush()

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore_path(event.src_path):
            return
        with self._lock:
            p = self._normalize_path(event.src_path)
            self._pending_modified.add(p)
            try:
                rel = p.resolve().relative_to(self.updater.root_dir).as_posix()
                self._pending_deleted.discard(rel)
            except ValueError:
                pass
            self._schedule_flush()

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore_path(event.src_path):
            return
        with self._lock:
            p = self._normalize_path(event.src_path)
            self._pending_modified.discard(p)
            try:
                rel = p.resolve().relative_to(self.updater.root_dir).as_posix()
                self._pending_deleted.add(rel)
            except ValueError:
                pass
            self._schedule_flush()

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        moved_event = event if isinstance(event, FileMovedEvent) else None
        if not moved_event:
            return

        with self._lock:
            # Handle source deletion
            if not self._should_ignore_path(moved_event.src_path):
                src_p = self._normalize_path(moved_event.src_path)
                self._pending_modified.discard(src_p)
                try:
                    rel_src = src_p.resolve().relative_to(self.updater.root_dir).as_posix()
                    self._pending_deleted.add(rel_src)
                except ValueError:
                    pass

            # Handle dest creation
            if not self._should_ignore_path(moved_event.dest_path):
                dest_p = self._normalize_path(moved_event.dest_path)
                self._pending_modified.add(dest_p)
                try:
                    rel_dest = dest_p.resolve().relative_to(self.updater.root_dir).as_posix()
                    self._pending_deleted.discard(rel_dest)
                except ValueError:
                    pass

            self._schedule_flush()



class RepoWatcher:
    """Manages Watchdog Observer thread for live background file watching."""

    def __init__(
        self,
        updater: IncrementalUpdater,
        debounce_seconds: float = 0.5,
        on_change_callback: Callable[[IncrementalSummary], None] | None = None,
    ) -> None:
        self.updater = updater
        self.debounce_seconds = debounce_seconds
        self.handler = DebouncedEventHandler(
            updater=updater,
            debounce_seconds=debounce_seconds,
            on_change_callback=on_change_callback,
        )
        self.observer = Observer()
        self._started = False

    def start(self) -> None:
        """Start the observer on the repository root directory."""
        if self._started:
            return
        self.observer.schedule(self.handler, str(self.updater.root_dir), recursive=True)
        self.observer.start()
        self._started = True

    def stop(self) -> None:
        """Stop the observer and wait for its worker thread to exit."""
        if not self._started:
            return
        self.observer.stop()
        self.observer.join(timeout=3.0)
        self._started = False

    def is_alive(self) -> bool:
        """Check if observer thread is running."""
        return self.observer.is_alive()


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is currently active."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            synchronize = 0x00100000
            process = kernel32.OpenProcess(synchronize, False, pid)
            if process != 0:
                kernel32.CloseHandle(process)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


def terminate_pid(pid: int) -> bool:
    """Terminate process by PID across platforms."""
    try:
        if os.name == "nt":
            res = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
            return res.returncode == 0
        else:
            os.kill(pid, signal.SIGTERM)
            return True
    except Exception:
        return False


class DaemonManager:
    """Controls the background indexing daemon lifecycle via PID file."""

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.pid_file = self.root_dir / DAEMON_PID_FILE
        self.log_file = self.root_dir / DAEMON_LOG_FILE

    def get_info(self) -> dict[str, Any] | None:
        """Read PID record from disk."""
        if not self.pid_file.is_file():
            return None
        try:
            data = json.loads(self.pid_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items()}
            return None
        except Exception:
            return None


    def status(self) -> dict[str, Any]:
        """Check current status of daemon."""
        info = self.get_info()
        if not info:
            return {"running": False, "pid": None}

        pid = info.get("pid")
        if pid and is_pid_alive(pid):
            uptime = round(time.time() - info.get("started_at", time.time()), 1)
            return {
                "running": True,
                "pid": pid,
                "started_at": info.get("started_at"),
                "uptime_seconds": uptime,
                "root_dir": info.get("root_dir", str(self.root_dir)),
                "log_file": str(self.log_file),
            }

        # Stale PID file detected, clean up
        self.pid_file.unlink(missing_ok=True)
        return {"running": False, "pid": None, "stale_cleaned": True}

    def start(self) -> int:
        """Start daemon in background detached process."""
        status = self.status()
        if status.get("running"):
            raise DaemonError(f"Indexing daemon is already running (PID {status['pid']}).")

        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "khaslana.cli.main",
            "watch",
            str(self.root_dir),
            "--daemon-mode",
        ]

        if os.name == "nt":
            # Windows detached process flags
            detached_flag = 0x00000008  # DETACHED_PROCESS
            new_group_flag = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            kwargs: dict[str, Any] = {
                "creationflags": detached_flag | new_group_flag,
                "cwd": str(self.root_dir),
            }
        else:
            kwargs = {
                "start_new_session": True,
                "cwd": str(self.root_dir),
            }

        with open(self.log_file, "a", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **kwargs,
            )

        info = {
            "pid": proc.pid,
            "started_at": time.time(),
            "root_dir": str(self.root_dir),
        }
        self.pid_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return proc.pid


    def stop(self) -> int:
        """Stop running daemon and clean up PID file."""
        info = self.get_info()
        if not info:
            raise DaemonError("No indexing daemon is currently running.")

        raw_pid = info.get("pid")
        if not raw_pid:
            self.pid_file.unlink(missing_ok=True)
            raise DaemonError("Daemon PID record was missing.")

        pid = int(raw_pid)
        if not is_pid_alive(pid):
            self.pid_file.unlink(missing_ok=True)
            raise DaemonError(f"Daemon process (PID {pid}) was not running.")

        terminate_pid(pid)
        self.pid_file.unlink(missing_ok=True)
        return pid

