"""Unit tests for Phase 5 Git integration, diff-aware review, and commit generator."""

import json
from pathlib import Path

import git
import pytest
from typer.testing import CliRunner

from khaslana.cli.main import app
from khaslana.core.config import KhaslanaConfig, LLMConfig
from khaslana.core.llm import FakeLLM, LLMRouter
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.models import CallEdge, ParsedFile, Symbol
from khaslana.intelligence.symbols import SymbolTable
from khaslana.tools.git import GitError, GitInspector
from khaslana.tools.review import (
    CodeReviewer,
    format_review_json,
    format_review_markdown,
)

runner = CliRunner()


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Fixture that initializes a temporary git repository with initial commit."""
    repo = git.Repo.init(tmp_path)
    # Configure user for git commits
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@khaslana.local").release()

    readme = tmp_path / "README.md"
    readme.write_text("# Test Repo\n", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")
    return tmp_path


def test_git_inspector_diffs_and_blame(temp_git_repo: Path) -> None:
    inspector = GitInspector(root_dir=temp_git_repo)
    repo = inspector.get_repo()

    # Modify existing file and create new file
    readme = temp_git_repo / "README.md"
    readme.write_text("# Test Repo\nUpdated line\n", encoding="utf-8")
    new_file = temp_git_repo / "main.py"
    new_file.write_text("def hello():\n    return 42\n", encoding="utf-8")

    # Unstaged diff
    unstaged_files = inspector.get_changed_files(staged=False)
    assert "README.md" in unstaged_files

    # Stage files
    repo.index.add(["README.md", "main.py"])

    staged_files = inspector.get_changed_files(staged=True)
    assert "README.md" in staged_files
    assert "main.py" in staged_files

    # Structured diff
    diffs = inspector.get_structured_diff(staged=True)
    assert len(diffs) >= 2
    main_diff = next(d for d in diffs if d.file == "main.py")
    assert main_diff.change_type in {"A", "M"}
    assert any("def hello():" in line for _, line in main_diff.added_lines)

    # Commit and test log and blame
    repo.index.commit("Add main.py and update readme")
    logs = inspector.get_log(max_count=2)
    assert len(logs) == 2
    assert "Add main.py" in logs[0]["summary"]

    blame = inspector.get_blame("README.md")
    assert len(blame) > 0


def test_git_not_a_repo(tmp_path: Path) -> None:
    non_repo = tmp_path / "not_git"
    non_repo.mkdir()
    inspector = GitInspector(root_dir=non_repo)
    with pytest.raises(GitError) as exc_info:
        inspector.get_repo()
    assert "not a git repository" in str(exc_info.value)


def test_enriched_diff_context(temp_git_repo: Path) -> None:
    db_file = temp_git_repo / ".khaslana" / "symbols.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    sym_table = SymbolTable(db_file)

    parsed = ParsedFile(
        file="auth.py",
        language="python",
        symbols=[
            Symbol(
                name="verify_token",
                symbol_type="function",
                file="auth.py",
                line_range=(1, 10),
                docstring="Verifies a token.",
            )
        ],
        imports=[],
        calls=[
            CallEdge(
                caller_file="api.py",
                caller_symbol="endpoint",
                callee_name="verify_token",
                line_number=20,
            )
        ],
    )
    sym_table.index_parsed_file(parsed)
    call_graph = CallGraph(sym_table)

    # Stage changes in auth.py
    auth_file = temp_git_repo / "auth.py"
    auth_file.write_text("def verify_token(tok):\n    return bool(tok)\n", encoding="utf-8")
    repo = git.Repo(temp_git_repo)
    repo.index.add(["auth.py"])

    inspector = GitInspector(root_dir=temp_git_repo)
    enriched = inspector.get_enriched_context(
        staged=True,
        symbol_table=sym_table,
        call_graph=call_graph,
    )

    assert len(enriched.diffs) == 1
    assert any(s.name == "verify_token" for s in enriched.touched_symbols)
    assert "verify_token" in enriched.callers
    assert "endpoint" in enriched.callers["verify_token"]


@pytest.mark.asyncio
async def test_code_reviewer_and_formatters(temp_git_repo: Path) -> None:
    # Stage a change in temp repo
    sample = temp_git_repo / "calc.py"
    sample.write_text("def div(a, b):\n    return a / b\n", encoding="utf-8")
    repo = git.Repo(temp_git_repo)
    repo.index.add(["calc.py"])

    # Script fake LLM response for review
    review_json = json.dumps(
        {
            "summary": "Potential division by zero issue detected.",
            "findings": [
                {
                    "file": "calc.py",
                    "line": 2,
                    "severity": "bug",
                    "title": "Division by zero",
                    "description": "b can be 0 causing ZeroDivisionError.",
                    "suggested_patch": "if b == 0: raise ValueError()",
                }
            ],
        }
    )

    fake_client = FakeLLM(
        config=LLMConfig(provider="fake"),
        responses=[review_json, "Summary: Adds calc.py", "feat(calc): add div function"],
    )
    router = LLMRouter(config=KhaslanaConfig(), custom_primary=fake_client)
    reviewer = CodeReviewer(router=router, root_dir=temp_git_repo)

    report = await reviewer.review(staged=True)
    assert report.total_files_analyzed == 1
    assert len(report.findings) == 1
    assert report.findings[0].severity == "bug"
    assert report.findings[0].file == "calc.py"

    # Test formatters
    md = format_review_markdown(report)
    assert "Division by zero" in md
    assert "### Suggested Patch" in md

    js = format_review_json(report)
    assert "ZeroDivisionError" in js

    # Explain diff
    explanation = await reviewer.explain_diff(staged=True)
    assert "Adds calc.py" in explanation

    # Commit message
    commit_msg = await reviewer.generate_commit_message()
    assert "feat(calc): add div function" in commit_msg


def test_cli_git_commands(temp_git_repo: Path) -> None:
    sample = temp_git_repo / "test_cli_file.py"
    sample.write_text("x = 10\n", encoding="utf-8")
    repo = git.Repo(temp_git_repo)
    repo.index.add(["test_cli_file.py"])

    # Review command (rich format)
    res_review = runner.invoke(
        app,
        ["review", "--staged", "--provider", "fake", "--path", str(temp_git_repo)],
    )
    assert res_review.exit_code == 0
    assert "Code Review Report" in res_review.output

    # Review command (json format)
    res_json = runner.invoke(
        app,
        [
            "review",
            "--staged",
            "--format",
            "json",
            "--provider",
            "fake",
            "--path",
            str(temp_git_repo),
        ],
    )
    assert res_json.exit_code == 0

    # Explain diff command
    res_explain = runner.invoke(
        app,
        ["explain-diff", "--staged", "--provider", "fake", "--path", str(temp_git_repo)],
    )
    assert res_explain.exit_code == 0

    # Commit msg command
    res_commit = runner.invoke(
        app,
        ["commit-msg", "--provider", "fake", "--path", str(temp_git_repo)],
    )
    assert res_commit.exit_code == 0
