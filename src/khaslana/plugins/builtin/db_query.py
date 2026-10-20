"""Built-in db-query plugin: read-only SQL database inspection."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from khaslana.plugins.models import Plugin, PluginMetadata
from khaslana.tools.base import Tool, ToolRegistry, ToolResult

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class DBQueryInput(BaseModel):
    """Input parameters for the read-only db_query tool."""

    connection_string: str = Field(
        description="Path to SQLite file (e.g., 'app.db' or 'sqlite:///app.db')"
    )
    query: str = Field(
        description="Read-only SQL query (SELECT, EXPLAIN, or non-mutating PRAGMA)"
    )
    max_rows: int = Field(default=50, description="Maximum number of rows to return")


class DBQueryTool(Tool):
    """Executes validated read-only SQL queries against SQLite databases."""

    name: str = "db_query"
    description: str = (
        "Execute read-only SQL queries (SELECT) against a local SQLite database file. "
        "Mutating statements (INSERT, UPDATE, DELETE, DROP) are strictly blocked."
    )
    input_schema: type[BaseModel] = DBQueryInput

    def execute(self, input: BaseModel) -> ToolResult:
        if not isinstance(input, DBQueryInput):
            return ToolResult(ok=False, output="", error="Invalid input model for db_query")

        query_clean = input.query.strip()

        # Enforce read-only safety
        if FORBIDDEN_SQL_PATTERN.search(query_clean):
            return ToolResult(
                ok=False,
                output="",
                error=(
                    "Mutating SQL command rejected. The 'db_query' tool only permits "
                    "read-only inspection queries (SELECT, EXPLAIN, PRAGMA table_info)."
                ),
            )

        # Parse SQLite path
        raw_conn = input.connection_string
        if raw_conn.startswith("sqlite:///"):
            db_path = Path(raw_conn[10:]).resolve()
        elif raw_conn.startswith("sqlite://"):
            db_path = Path(raw_conn[9:]).resolve()
        elif raw_conn.startswith(("postgres://", "postgresql://")):
            return ToolResult(
                ok=False,
                output="",
                error=(
                    "PostgreSQL query execution requires 'psycopg2' or 'asyncpg' library, "
                    "which is not currently configured."
                ),
            )
        else:
            db_path = Path(raw_conn).resolve()

        if not db_path.is_file():
            return ToolResult(
                ok=False,
                output="",
                error=f"Database file not found: {db_path}",
            )

        try:
            # Connect in read-only mode using URI
            uri = f"file:{db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(query_clean)
            rows = cursor.fetchmany(input.max_rows)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()

            if not columns:
                return ToolResult(
                    ok=True,
                    output="Query executed successfully with 0 columns returned.",
                    metadata={"row_count": 0},
                )

            # Format markdown table
            headers = " | ".join(str(c) for c in columns)
            separators = " | ".join("---" for _ in columns)
            row_lines: list[str] = []
            for row in rows:
                row_str = " | ".join(str(row[c]) if row[c] is not None else "NULL" for c in columns)
                row_lines.append(f"| {row_str} |")

            table_output = f"| {headers} |\n| {separators} |\n" + "\n".join(row_lines)
            if len(rows) == input.max_rows:
                table_output += f"\n\n*(Results capped at {input.max_rows} rows)*"

            return ToolResult(
                ok=True,
                output=table_output,
                metadata={"row_count": len(rows), "columns": columns},
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                output="",
                error=f"SQL query error: {e}",
            )


class DBQueryPlugin(Plugin):
    """Plugin providing database query capabilities."""

    metadata = PluginMetadata(
        name="db-query",
        version="0.1.0",
        description="Read-only SQLite database inspection tool",
        author="Khaslana Team",
        enabled=True,
        tools=["db_query"],
        source="builtin",
    )

    def register(self, registry: ToolRegistry) -> None:
        registry.register(DBQueryTool())
