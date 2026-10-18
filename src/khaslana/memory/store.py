"""SQLite-backed persistent store for interactions, fixes, feedback, and style profiles."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from khaslana.core.errors import MemoryStoreError
from khaslana.memory.models import (
    FeedbackRecord,
    FixRecord,
    MemorySearchResult,
    QueryRecord,
    StyleProfile,
)

if TYPE_CHECKING:
    from khaslana.retrieval.embeddings import EmbeddingProvider


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class MemoryStore:
    """SQLite-backed memory store with semantic vector search for past interactions."""

    def __init__(
        self,
        db_path: Path | str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return an active database connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create necessary database tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queries (
                        id TEXT PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        response_text TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        session_id TEXT,
                        model TEXT,
                        provider TEXT,
                        tokens_used INTEGER,
                        embedding_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fixes (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        original_code TEXT NOT NULL,
                        fixed_code TEXT NOT NULL,
                        diff TEXT NOT NULL,
                        status TEXT NOT NULL,
                        test_cmd TEXT,
                        attempts_count INTEGER NOT NULL,
                        timestamp REAL NOT NULL,
                        embedding_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        interaction_id TEXT NOT NULL,
                        interaction_type TEXT NOT NULL,
                        rating TEXT NOT NULL,
                        notes TEXT,
                        timestamp REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS style_profile (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        profile_json TEXT NOT NULL,
                        last_updated REAL NOT NULL
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to initialize memory database at {self.db_path}: {exc}"
            ) from exc

    def record_query(
        self,
        query_text: str,
        response_text: str,
        session_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        tokens_used: int | None = None,
    ) -> QueryRecord:
        """Save a user query and assistant response."""
        record = QueryRecord(
            query_text=query_text,
            response_text=response_text,
            session_id=session_id,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
        )

        embedding_json = None
        if self.embedding_provider:
            try:
                emb = self.embedding_provider.encode_query(query_text)
                embedding_json = json.dumps(emb)
            except Exception:
                pass

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO queries (
                    id, query_text, response_text, timestamp, session_id,
                    model, provider, tokens_used, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.query_text,
                    record.response_text,
                    record.timestamp,
                    record.session_id,
                    record.model,
                    record.provider,
                    record.tokens_used,
                    embedding_json,
                ),
            )
            conn.commit()
        return record

    def record_fix(
        self,
        file_path: str,
        original_code: str,
        fixed_code: str,
        diff: str,
        status: str,
        test_cmd: str | None = None,
        attempts_count: int = 1,
    ) -> FixRecord:
        """Save an accepted or rejected code fix attempt."""
        record = FixRecord(
            file_path=file_path,
            original_code=original_code,
            fixed_code=fixed_code,
            diff=diff,
            status="accepted" if status == "accepted" else "rejected",
            test_cmd=test_cmd,
            attempts_count=attempts_count,
        )

        embedding_json = None
        if self.embedding_provider:
            try:
                emb_text = f"Fix for {file_path}:\n{diff[:500]}"
                emb = self.embedding_provider.encode_query(emb_text)
                embedding_json = json.dumps(emb)
            except Exception:
                pass

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO fixes (
                    id, file_path, original_code, fixed_code, diff,
                    status, test_cmd, attempts_count, timestamp, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.file_path,
                    record.original_code,
                    record.fixed_code,
                    record.diff,
                    record.status,
                    record.test_cmd,
                    record.attempts_count,
                    record.timestamp,
                    embedding_json,
                ),
            )
            conn.commit()
        return record

    def record_feedback(
        self,
        interaction_id: str,
        interaction_type: str,
        rating: str,
        notes: str | None = None,
    ) -> FeedbackRecord:
        """Record user feedback ('good' or 'bad') for an interaction."""
        record = FeedbackRecord(
            interaction_id=interaction_id,
            interaction_type="fix" if interaction_type == "fix" else "query",
            rating="bad" if rating == "bad" else "good",
            notes=notes,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    id, interaction_id, interaction_type, rating, notes, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.interaction_id,
                    record.interaction_type,
                    record.rating,
                    record.notes,
                    record.timestamp,
                ),
            )
            conn.commit()
        return record

    def search_interactions(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        """Perform semantic or keyword search across stored queries and fixes."""
        query_emb: list[float] | None = None
        if self.embedding_provider:
            try:
                query_emb = self.embedding_provider.encode_query(query)
            except Exception:
                query_emb = None

        results: list[MemorySearchResult] = []

        with self._get_connection() as conn:
            # Query table search
            query_rows = conn.execute(
                "SELECT id, query_text, response_text, timestamp, embedding_json FROM queries"
            ).fetchall()

            for row in query_rows:
                q_text = row["query_text"]
                r_text = row["response_text"]
                emb_json = row["embedding_json"]
                score = 0.0

                if query_emb and emb_json:
                    try:
                        emb = json.loads(emb_json)
                        score = cosine_similarity(query_emb, emb)
                    except Exception:
                        score = 0.0
                elif query.lower() in q_text.lower() or query.lower() in r_text.lower():
                    score = 0.6

                if score > 0.1:
                    snippet = f"Q: {q_text}\nA: {r_text[:300]}"
                    results.append(
                        MemorySearchResult(
                            record_id=row["id"],
                            record_type="query",
                            snippet=snippet,
                            similarity=score,
                            timestamp=row["timestamp"],
                            metadata={"query": q_text, "response": r_text[:500]},
                        )
                    )

            # Fixes table search
            fix_rows = conn.execute(
                "SELECT id, file_path, diff, status, timestamp, embedding_json FROM fixes"
            ).fetchall()

            for row in fix_rows:
                f_path = row["file_path"]
                f_diff = row["diff"]
                f_status = row["status"]
                emb_json = row["embedding_json"]
                score = 0.0

                if query_emb and emb_json:
                    try:
                        emb = json.loads(emb_json)
                        score = cosine_similarity(query_emb, emb)
                    except Exception:
                        score = 0.0
                elif query.lower() in f_path.lower() or query.lower() in f_diff.lower():
                    score = 0.55

                if score > 0.1:
                    snippet = f"Fix ({f_status}) for {f_path}:\n{f_diff[:300]}"
                    results.append(
                        MemorySearchResult(
                            record_id=row["id"],
                            record_type="fix",
                            snippet=snippet,
                            similarity=score,
                            timestamp=row["timestamp"],
                            metadata={"file_path": f_path, "status": f_status},
                        )
                    )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def list_queries(self, limit: int = 20) -> list[QueryRecord]:
        """List most recent user queries."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, query_text, response_text, timestamp, session_id,
                       model, provider, tokens_used
                FROM queries
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                QueryRecord(
                    id=row["id"],
                    query_text=row["query_text"],
                    response_text=row["response_text"],
                    timestamp=row["timestamp"],
                    session_id=row["session_id"],
                    model=row["model"],
                    provider=row["provider"],
                    tokens_used=row["tokens_used"],
                )
                for row in rows
            ]

    def list_fixes(self, limit: int = 20) -> list[FixRecord]:
        """List most recent code fixes."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, file_path, original_code, fixed_code, diff,
                       status, test_cmd, attempts_count, timestamp
                FROM fixes
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                FixRecord(
                    id=row["id"],
                    file_path=row["file_path"],
                    original_code=row["original_code"],
                    fixed_code=row["fixed_code"],
                    diff=row["diff"],
                    status="accepted" if row["status"] == "accepted" else "rejected",
                    test_cmd=row["test_cmd"],
                    attempts_count=row["attempts_count"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    def get_latest_interaction_id(self) -> str | None:
        """Fetch the most recent interaction ID across queries and fixes."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM (
                    SELECT id, timestamp FROM queries
                    UNION ALL
                    SELECT id, timestamp FROM fixes
                )
                ORDER BY timestamp DESC
                LIMIT 1
                """
            ).fetchone()
            return str(row["id"]) if row else None

    def save_style_profile(self, profile: StyleProfile) -> None:
        """Persist user style profile to SQLite."""
        profile_json = profile.model_dump_json()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO style_profile (id, profile_json, last_updated)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    last_updated = excluded.last_updated
                """,
                (profile_json, profile.last_updated),
            )
            conn.commit()

    def load_style_profile(self) -> StyleProfile | None:
        """Load stored user style profile if available."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT profile_json FROM style_profile WHERE id = 1"
            ).fetchone()
            if row:
                try:
                    data = json.loads(row["profile_json"])
                    return StyleProfile.model_validate(data)
                except Exception:
                    return None
        return None

    def forget(self, record_id: str) -> bool:
        """Delete an interaction record by its ID from queries, fixes, and feedback."""
        deleted = False
        with self._get_connection() as conn:
            c1 = conn.execute("DELETE FROM queries WHERE id = ?", (record_id,)).rowcount
            c2 = conn.execute("DELETE FROM fixes WHERE id = ?", (record_id,)).rowcount
            c3 = conn.execute(
                "DELETE FROM feedback WHERE id = ? OR interaction_id = ?",
                (record_id, record_id),
            ).rowcount
            conn.commit()
            if c1 > 0 or c2 > 0 or c3 > 0:
                deleted = True
        return deleted

    def clear(self) -> None:
        """Clear all stored interaction history and style profiles."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM queries")
            conn.execute("DELETE FROM fixes")
            conn.execute("DELETE FROM feedback")
            conn.execute("DELETE FROM style_profile")
            conn.commit()

    def export_data(self) -> dict[str, Any]:
        """Export all stored memory data as a structured dictionary."""
        queries = [q.model_dump() for q in self.list_queries(limit=1000)]
        fixes = [f.model_dump() for f in self.list_fixes(limit=1000)]
        profile = self.load_style_profile()

        with self._get_connection() as conn:
            feedback_rows = conn.execute(
                "SELECT * FROM feedback ORDER BY timestamp DESC"
            ).fetchall()
            feedback = [dict(r) for r in feedback_rows]

        return {
            "queries": queries,
            "fixes": fixes,
            "feedback": feedback,
            "style_profile": profile.model_dump() if profile else None,
        }
