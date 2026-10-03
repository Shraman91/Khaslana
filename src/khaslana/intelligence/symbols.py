"""SQLite-backed symbol table indexing definitions, imports, and call sites."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from khaslana.intelligence.models import CallEdge, ParsedFile, Symbol


class SymbolTable:
    """Persistent SQLite database for fast symbol lookups, definitions, and usage sites."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    file TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    docstring TEXT,
                    parent_symbol TEXT,
                    signature TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file TEXT NOT NULL,
                    imported_module TEXT NOT NULL,
                    imported_symbol TEXT,
                    alias TEXT,
                    line_number INTEGER NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    caller_file TEXT NOT NULL,
                    caller_symbol TEXT NOT NULL,
                    callee_name TEXT NOT NULL,
                    line_number INTEGER NOT NULL
                );
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name);")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file);")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_symbol);"
            )

    def index_parsed_file(self, parsed: ParsedFile) -> None:
        """Add or replace all symbols, imports, and call sites for a file."""
        with self._conn:
            # Delete existing records for this file
            self._conn.execute("DELETE FROM symbols WHERE file = ?", (parsed.file,))
            self._conn.execute("DELETE FROM imports WHERE file = ?", (parsed.file,))
            self._conn.execute("DELETE FROM calls WHERE caller_file = ?", (parsed.file,))

            # Insert symbols
            for sym in parsed.symbols:
                self._conn.execute(
                    """
                    INSERT INTO symbols (
                        name, symbol_type, file, start_line, end_line,
                        docstring, parent_symbol, signature
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sym.name,
                        sym.symbol_type,
                        sym.file,
                        sym.line_range[0],
                        sym.line_range[1],
                        sym.docstring,
                        sym.parent_symbol,
                        sym.signature,
                    ),
                )

            # Insert imports
            for imp in parsed.imports:
                self._conn.execute(
                    """
                    INSERT INTO imports (file, imported_module, imported_symbol, alias, line_number)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        imp.file,
                        imp.imported_module,
                        imp.imported_symbol,
                        imp.alias,
                        imp.line_number,
                    ),
                )

            # Insert calls
            for call in parsed.calls:
                self._conn.execute(
                    """
                    INSERT INTO calls (caller_file, caller_symbol, callee_name, line_number)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        call.caller_file,
                        call.caller_symbol,
                        call.callee_name,
                        call.line_number,
                    ),
                )

    def find_definitions(self, symbol_name: str) -> list[Symbol]:
        """Find definition sites for a symbol name (exact or parent.child)."""
        clean_name = symbol_name.split(".")[-1]
        cur = self._conn.execute(
            """
            SELECT
                name, symbol_type, file, start_line, end_line,
                docstring, parent_symbol, signature
            FROM symbols
            WHERE name = ? OR (parent_symbol || '.' || name) = ?
            """,
            (clean_name, symbol_name),
        )
        return [
            Symbol(
                name=row["name"],
                symbol_type=row["symbol_type"],
                file=row["file"],
                line_range=(row["start_line"], row["end_line"]),
                docstring=row["docstring"],
                parent_symbol=row["parent_symbol"],
                signature=row["signature"],
            )
            for row in cur.fetchall()
        ]

    def find_references(self, symbol_name: str) -> list[CallEdge]:
        """Find all call sites referencing the symbol name."""
        clean_name = symbol_name.split(".")[-1]
        cur = self._conn.execute(
            """
            SELECT caller_file, caller_symbol, callee_name, line_number
            FROM calls
            WHERE callee_name = ?
            """,
            (clean_name,),
        )
        return [
            CallEdge(
                caller_file=row["caller_file"],
                caller_symbol=row["caller_symbol"],
                callee_name=row["callee_name"],
                line_number=row["line_number"],
            )
            for row in cur.fetchall()
        ]

    def get_symbols_in_file(self, file_path: str) -> list[Symbol]:
        """Retrieve all symbols defined in a file."""
        cur = self._conn.execute(
            """
            SELECT
                name, symbol_type, file, start_line, end_line,
                docstring, parent_symbol, signature
            FROM symbols
            WHERE file = ?
            ORDER BY start_line ASC
            """,
            (file_path,),
        )
        return [
            Symbol(
                name=row["name"],
                symbol_type=row["symbol_type"],
                file=row["file"],
                line_range=(row["start_line"], row["end_line"]),
                docstring=row["docstring"],
                parent_symbol=row["parent_symbol"],
                signature=row["signature"],
            )
            for row in cur.fetchall()
        ]

    def get_all_symbols(self) -> list[Symbol]:
        """Retrieve all indexed symbols across the repository."""
        cur = self._conn.execute(
            """
            SELECT
                name, symbol_type, file, start_line, end_line,
                docstring, parent_symbol, signature
            FROM symbols
            """
        )
        return [
            Symbol(
                name=row["name"],
                symbol_type=row["symbol_type"],
                file=row["file"],
                line_range=(row["start_line"], row["end_line"]),
                docstring=row["docstring"],
                parent_symbol=row["parent_symbol"],
                signature=row["signature"],
            )
            for row in cur.fetchall()
        ]

    def count_symbols(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM symbols")
        return int(cur.fetchone()[0])

    def count_calls(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM calls")
        return int(cur.fetchone()[0])

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM symbols")
            self._conn.execute("DELETE FROM imports")
            self._conn.execute("DELETE FROM calls")

    def close(self) -> None:
        self._conn.close()
