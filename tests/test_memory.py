"""Unit and integration tests for Phase 7 Memory System and Style Learner."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from khaslana.cli.main import app
from khaslana.memory.models import FixRecord, StyleProfile
from khaslana.memory.store import MemoryStore, cosine_similarity
from khaslana.memory.style import StyleLearner
from khaslana.retrieval.embeddings import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for memory vector tests."""

    @property
    def dimension(self) -> int:
        return 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            val_api = 1.0 if "api" in text.lower() or "auth" in text.lower() else 0.0
            val_db = 1.0 if "db" in text.lower() or "sql" in text.lower() else 0.0
            val_fix = 1.0 if "fix" in text.lower() or "bug" in text.lower() else 0.0
            vectors.append([val_api, val_db, val_fix, 0.5])
        return vectors


def test_cosine_similarity() -> None:
    """Test cosine similarity calculation with identical, orthogonal, and empty vectors."""
    assert cosine_similarity([], []) == 0.0
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    assert cosine_similarity([1.0, 1.0], [1.0, 1.0]) > 0.99


def test_memory_store_crud(tmp_path: Path) -> None:
    """Test recording and retrieving queries, fixes, and feedback."""
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_file)

    q1 = store.record_query(
        query_text="How does auth work?",
        response_text="Auth uses AuthService with JWT tokens.",
        model="fake-model",
        provider="fake",
    )
    assert q1.id
    assert q1.query_text == "How does auth work?"

    f1 = store.record_fix(
        file_path="src/auth.py",
        original_code="return False",
        fixed_code="return True",
        diff="+ return True\n- return False",
        status="accepted",
        test_cmd="pytest",
        attempts_count=2,
    )
    assert f1.id
    assert f1.status == "accepted"

    fb1 = store.record_feedback(
        interaction_id=q1.id,
        interaction_type="query",
        rating="good",
        notes="Very clear explanation",
    )
    assert fb1.id
    assert fb1.rating == "good"

    queries = store.list_queries(limit=10)
    assert len(queries) == 1
    assert queries[0].id == q1.id

    fixes = store.list_fixes(limit=10)
    assert len(fixes) == 1
    assert fixes[0].id == f1.id

    # Test forget
    assert store.forget(q1.id)
    assert len(store.list_queries()) == 0

    # Test clear
    store.clear()
    assert len(store.list_fixes()) == 0


def test_memory_store_semantic_search(tmp_path: Path) -> None:
    """Test vector similarity search over interactions."""
    db_file = tmp_path / "memory.db"
    embedder = FakeEmbeddingProvider()
    store = MemoryStore(db_path=db_file, embedding_provider=embedder)

    q1 = store.record_query(
        query_text="Explain database connection pooling",
        response_text="Database connections are pooled using SQLAlchemy.",
    )
    _q2 = store.record_query(
        query_text="How to authenticate API routes?",
        response_text="Use OAuth2 bearer tokens in route dependencies.",
    )
    f1 = store.record_fix(
        file_path="src/db.py",
        original_code="bad sql",
        fixed_code="good sql",
        diff="+ good sql",
        status="accepted",
    )

    results = store.search_interactions("database sql query", limit=5)
    assert len(results) >= 2
    # The database query and db fix should rank at the top
    top_ids = [r.record_id for r in results[:2]]
    assert q1.id in top_ids or f1.id in top_ids


def test_style_learner_analysis(tmp_path: Path) -> None:
    """Test StyleLearner analyzing python files for naming, typing, and docstrings."""
    sample_file = tmp_path / "sample.py"
    sample_code = '''"""Sample module for testing style learner."""

from pydantic import BaseModel
from typer import Typer

app = Typer()


class UserProfile(BaseModel):
    """User profile data model."""
    username: str
    is_active: bool = True


def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax for given amount.

    Args:
        amount: Pre-tax total.
        rate: Tax percentage.

    Returns:
        Computed tax.
    """
    return amount * rate
'''
    sample_file.write_text(sample_code, encoding="utf-8")

    learner = StyleLearner()
    profile = learner.analyze_codebase(tmp_path)

    assert "snake_case" in profile.naming_convention
    assert (
        "Strict type hints" in profile.typing_discipline
        or "type hints" in profile.typing_discipline
    )
    assert "Google-style" in profile.docstring_style or "docstrings" in profile.docstring_style
    assert any("Pydantic" in p for p in profile.framework_patterns)
    assert any("Typer" in p for p in profile.framework_patterns)

    prompt_context = profile.to_prompt_context()
    assert "### User Coding Style Profile" in prompt_context
    assert "Naming:" in prompt_context
    assert "Typing:" in prompt_context


def test_style_learner_update_from_fix() -> None:
    """Test style profile refinement after accepted fix."""
    learner = StyleLearner()
    init_profile = StyleProfile()

    fix = FixRecord(
        file_path="service.py",
        original_code="def run(): pass",
        fixed_code="def run() -> bool:\n    return True\n",
        diff="+ return True",
        status="accepted",
    )

    updated = learner.update_from_accepted_fix(fix, init_profile)
    assert updated is not None
    assert updated.last_updated >= init_profile.last_updated


def test_cli_memory_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI commands: memory list, search, export, clear, and feedback."""
    runner = CliRunner()
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_file)
    q1 = store.record_query("How to write CLI apps?", "Use Typer with Annotated.")

    def mock_get_store() -> MemoryStore:
        return store

    monkeypatch.setattr("khaslana.cli.memory_cmd._get_memory_store", mock_get_store)

    # Test list
    res_list = runner.invoke(app, ["memory", "list"])
    assert res_list.exit_code == 0
    assert q1.id in res_list.output or "CLI apps" in res_list.output

    # Test search
    res_search = runner.invoke(app, ["memory", "search", "CLI apps"])
    assert res_search.exit_code == 0
    assert q1.id in res_search.output

    # Test feedback good
    res_fb = runner.invoke(app, ["feedback", "good", q1.id, "--notes", "Great!"])
    assert res_fb.exit_code == 0
    assert "GOOD" in res_fb.output

    # Test export
    res_exp = runner.invoke(app, ["memory", "export"])
    assert res_exp.exit_code == 0
    assert "queries" in res_exp.output

    # Test forget
    res_forget = runner.invoke(app, ["memory", "forget", q1.id])
    assert res_forget.exit_code == 0
    assert "removed" in res_forget.output

    # Test clear with --yes
    res_clear = runner.invoke(app, ["memory", "clear", "--yes"])
    assert res_clear.exit_code == 0
    assert "cleared" in res_clear.output
