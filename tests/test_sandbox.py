"""Unit and integration tests for Sandbox execution and self-correcting fix loop."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from khaslana.cli.main import app
from khaslana.core.errors import SandboxUnavailableError
from khaslana.core.llm import FakeLLM
from khaslana.sandbox.docker_runner import (
    DockerRunner,
    FakeSandboxRunner,
    detect_image,
)
from khaslana.sandbox.fix_loop import (
    SelfCorrectingLoop,
    auto_detect_test_cmd,
    compute_unified_diff,
    extract_code_fix,
)
from khaslana.sandbox.models import ExecutionResult


def test_detect_image(tmp_path: Path) -> None:
    """Test image detection from file extensions and workspace markers."""
    py_file = tmp_path / "main.py"
    py_file.write_text("print('hi')", encoding="utf-8")
    assert detect_image(tmp_path, py_file) == "python:3.11-slim"

    ts_file = tmp_path / "app.ts"
    ts_file.write_text("console.log('hi');", encoding="utf-8")
    assert detect_image(tmp_path, ts_file) == "node:20-slim"

    go_file = tmp_path / "main.go"
    go_file.write_text("package main", encoding="utf-8")
    assert detect_image(tmp_path, go_file) == "golang:1.22-alpine"

    rs_file = tmp_path / "lib.rs"
    rs_file.write_text("fn main() {}", encoding="utf-8")
    assert detect_image(tmp_path, rs_file) == "rust:1.77-slim"

    java_file = tmp_path / "App.java"
    java_file.write_text("class App {}", encoding="utf-8")
    assert detect_image(tmp_path, java_file) == "eclipse-temurin:17-jdk-jammy"


def test_auto_detect_test_cmd(tmp_path: Path) -> None:
    """Test test command inference for various project structures."""
    py_file = tmp_path / "test_math.py"
    py_file.write_text("def test_ok(): pass", encoding="utf-8")
    assert "pytest" in auto_detect_test_cmd(tmp_path, py_file)

    pkg_json = tmp_path / "package.json"
    pkg_json.write_text("{}", encoding="utf-8")
    assert auto_detect_test_cmd(tmp_path, None) == "npm test"


def test_extract_code_fix() -> None:
    """Test extracting code from markdown blocks and JSON responses."""
    md_response = """Here is the fix:
```python
def add(a: int, b: int) -> int:
    return a + b
```
Hope this helps!"""
    code, _ = extract_code_fix(md_response)
    assert "return a + b" in code
    assert "add(a: int" in code

    json_response = (
        '{"explanation": "Fixed operator", "fixed_code": "def sub(a, b): return a - b"}'
    )
    code_j, expl_j = extract_code_fix(json_response)
    assert "return a - b" in code_j
    assert expl_j == "Fixed operator"


def test_compute_unified_diff() -> None:
    """Test unified diff calculation."""
    orig = "def compute():\n    return 1\n"
    mod = "def compute():\n    return 2\n"
    diff = compute_unified_diff(orig, mod, "calc.py")
    assert "-    return 1" in diff
    assert "+    return 2" in diff
    assert "a/calc.py" in diff


def test_docker_runner_unavailable() -> None:
    """Test DockerRunner raises SandboxUnavailableError with actionable hint when ping fails."""
    mock_client = MagicMock()
    mock_client.ping.side_effect = Exception(
        "Connection refused on npipe:////./pipe/docker_engine"
    )
    runner = DockerRunner(client=mock_client)
    runner._client = None

    with pytest.raises(SandboxUnavailableError) as exc_info:
        # Force fresh client retrieval
        runner._get_client()

    assert "Docker is not available" in str(exc_info.value)
    assert exc_info.value.hint is not None
    assert "Docker Desktop" in exc_info.value.hint


def test_docker_runner_execution_with_mock(tmp_path: Path) -> None:
    """Test DockerRunner container creation, execution, and cleanup."""
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.side_effect = (
        lambda stdout=False, stderr=False: b"Tests passed\n" if stdout else b""
    )
    mock_client.containers.create.return_value = mock_container

    runner = DockerRunner(client=mock_client, default_timeout=10)
    res = runner.run_command("pytest", workspace_dir=tmp_path)

    assert res.is_success
    assert res.exit_code == 0
    assert "Tests passed" in res.stdout
    mock_container.start.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_docker_runner_timeout(tmp_path: Path) -> None:
    """Test DockerRunner timeout handling."""
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.side_effect = Exception("Read timed out")
    mock_client.containers.create.return_value = mock_container

    runner = DockerRunner(client=mock_client, default_timeout=5)
    res = runner.run_command("sleep 100", workspace_dir=tmp_path)

    assert not res.is_success
    assert res.timed_out
    assert res.exit_code == 124
    mock_container.kill.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_self_correcting_loop_already_passing(tmp_path: Path) -> None:
    """Test fix loop when tests already pass."""
    file_path = tmp_path / "math_mod.py"
    file_path.write_text("def mul(a, b): return a * b\n", encoding="utf-8")

    runner = FakeSandboxRunner(canned_results=[
        ExecutionResult(exit_code=0, stdout="1 passed", stderr="", duration_seconds=0.1)
    ])
    llm = FakeLLM(responses=[])

    loop = SelfCorrectingLoop(
        runner=runner,
        llm=llm,
        project_root=tmp_path,
        file_rel_path="math_mod.py",
        test_cmd="pytest",
        max_attempts=3,
    )
    res = await loop.run()

    assert res.success
    assert len(res.attempts) == 0
    assert file_path.read_text(encoding="utf-8") == "def mul(a, b): return a * b\n"


@pytest.mark.asyncio
async def test_self_correcting_loop_success_on_attempt_2(tmp_path: Path) -> None:
    """Test self-correcting loop failing on attempt 1 and succeeding on attempt 2."""
    file_path = tmp_path / "service.py"
    orig_code = "def divide(a, b):\n    return a * b  # bug\n"
    file_path.write_text(orig_code, encoding="utf-8")

    canned_results = [
        # Baseline check (fails)
        ExecutionResult(exit_code=1, stdout="", stderr="AssertionError: 2 != 0.5"),
        # Attempt 1 run (fails)
        ExecutionResult(exit_code=1, stdout="", stderr="ZeroDivisionError: division by zero"),
        # Attempt 2 run (passes)
        ExecutionResult(exit_code=0, stdout="1 passed in 0.01s", stderr=""),
    ]
    runner = FakeSandboxRunner(canned_results=canned_results)

    fixes = [
        "```python\ndef divide(a, b):\n    return a / b\n```",
        "```python\ndef divide(a, b):\n    if b == 0:\n        return 0.0\n    return a / b\n```",
    ]
    llm = FakeLLM(responses=fixes)

    loop = SelfCorrectingLoop(
        runner=runner,
        llm=llm,
        project_root=tmp_path,
        file_rel_path="service.py",
        test_cmd="pytest",
        max_attempts=3,
    )
    res = await loop.run()

    assert res.success
    assert len(res.attempts) == 2
    assert not res.attempts[0].success
    assert res.attempts[1].success
    assert "if b == 0" in (res.final_code or "")
    # Verify that original working tree file was NEVER touched during loop
    assert file_path.read_text(encoding="utf-8") == orig_code


@pytest.mark.asyncio
async def test_self_correcting_loop_exhausted_attempts(tmp_path: Path) -> None:
    """Test self-correcting loop when all attempts fail."""
    file_path = tmp_path / "broken.py"
    orig_code = "syntax error here"
    file_path.write_text(orig_code, encoding="utf-8")

    canned_results = [
        ExecutionResult(exit_code=1, stdout="", stderr="SyntaxError"),
        ExecutionResult(exit_code=1, stdout="", stderr="SyntaxError still"),
        ExecutionResult(exit_code=1, stdout="", stderr="SyntaxError again"),
    ]
    runner = FakeSandboxRunner(canned_results=canned_results)
    llm = FakeLLM(responses=["bad code 1", "bad code 2"])

    loop = SelfCorrectingLoop(
        runner=runner,
        llm=llm,
        project_root=tmp_path,
        file_rel_path="broken.py",
        test_cmd="pytest",
        max_attempts=2,
    )
    res = await loop.run()

    assert not res.success
    assert len(res.attempts) == 2
    assert res.final_code is None
    # Real file untouched
    assert file_path.read_text(encoding="utf-8") == orig_code


def test_cli_fix_command_docker_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test CLI khaslana fix fails gracefully without raw traceback when Docker is down."""
    runner = CliRunner()
    target_file = tmp_path / "app.py"
    target_file.write_text("print(1)", encoding="utf-8")

    def mock_get_client(self: object) -> None:
        raise SandboxUnavailableError(
            message="Docker is not available: daemon down",
            hint="Please install Docker and ensure Docker Desktop is running.",
        )

    monkeypatch.setattr(DockerRunner, "_get_client", mock_get_client)

    result = runner.invoke(app, ["fix", str(target_file)])
    assert result.exit_code == 1
    assert "Docker is not available" in result.output
    assert "Hint: Please install Docker and ensure Docker Desktop is running." in result.output
    assert "Traceback" not in result.output
