"""Unit tests for Phase 4 Agent ReAct loop, parser, and commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from khaslana.agents.loop import AgentLoop
from khaslana.agents.models import AgentAction, AgentFinish
from khaslana.agents.parser import parse_agent_output
from khaslana.cli.main import app
from khaslana.core.config import LLMConfig
from khaslana.core.llm import FakeLLM
from khaslana.tools.base import ToolRegistry
from khaslana.tools.core import ReadFileTool

runner = CliRunner()


def test_parser_valid_json() -> None:
    action_json = (
        '{"thought": "Need to check file", "action": "read_file", '
        '"action_input": {"file_path": "main.py"}}'
    )
    res = parse_agent_output(action_json)
    assert isinstance(res, AgentAction)
    assert res.action == "read_file"
    assert res.action_input["file_path"] == "main.py"

    finish_json = '{"thought": "Done analyzing", "final_answer": "Bug fixed in line 42"}'
    res_fin = parse_agent_output(finish_json)
    assert isinstance(res_fin, AgentFinish)
    assert res_fin.final_answer == "Bug fixed in line 42"


def test_parser_repair_markdown_and_trailing_commas() -> None:
    markdown_wrapped = """Here is what I plan to do:
```json
{
  "thought": "Reading config",
  "action": "read_file",
  "action_input": {
    "file_path": "config.toml",
  },
}
```
"""
    res = parse_agent_output(markdown_wrapped)
    assert isinstance(res, AgentAction)
    assert res.action == "read_file"
    assert res.action_input["file_path"] == "config.toml"


def test_parser_fallback_direct_answer() -> None:
    raw_text = "The issue is caused by a missing import on line 5."
    res = parse_agent_output(raw_text)
    assert isinstance(res, AgentFinish)
    assert "missing import" in res.final_answer


@pytest.mark.asyncio
async def test_agent_loop_deterministic_execution(tmp_path: Path) -> None:
    target_file = tmp_path / "hello.py"
    target_file.write_text("print('hello world')\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(ReadFileTool(root_dir=tmp_path))

    # Scripted responses for FakeLLM:
    # 1. Action: read hello.py
    # 2. Finish: final answer
    resp1 = json.dumps(
        {
            "thought": "I should read hello.py to see what it prints.",
            "action": "read_file",
            "action_input": {"file_path": "hello.py"},
        }
    )
    resp2 = json.dumps(
        {
            "thought": "I have read hello.py and confirmed the content.",
            "final_answer": "The file hello.py prints 'hello world'.",
        }
    )

    fake_llm = FakeLLM(config=LLMConfig(provider="fake"), responses=[resp1, resp2])
    loop = AgentLoop(
        llm=fake_llm,
        tool_registry=registry,
        max_iterations=5,
        verbose=False,
    )

    trace = await loop.run("Inspect hello.py")
    assert trace.success is True
    assert len(trace.steps) == 2
    assert trace.steps[0].action == "read_file"
    assert "hello world" in trace.steps[0].observation
    assert trace.final_answer == "The file hello.py prints 'hello world'."


@pytest.mark.asyncio
async def test_agent_loop_repeat_detection(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool(root_dir=tmp_path))

    # Repeat identical action 4 times
    repeat_action = json.dumps(
        {
            "thought": "Repeatedly trying to read missing file",
            "action": "read_file",
            "action_input": {"file_path": "missing.py"},
        }
    )
    fake_llm = FakeLLM(
        config=LLMConfig(provider="fake"),
        responses=[repeat_action, repeat_action, repeat_action, repeat_action],
    )
    loop = AgentLoop(
        llm=fake_llm,
        tool_registry=registry,
        max_iterations=10,
        verbose=False,
    )

    trace = await loop.run("Fix bug")
    assert trace.success is False
    assert trace.final_answer and "detected repeated tool calls" in trace.final_answer


def test_debug_and_analyze_cli_commands(tmp_path: Path) -> None:
    log_file = tmp_path / "error.log"
    log_file.write_text("Traceback (most recent call last):\n  ZeroDivisionError: division by zero")

    # Run debug command with fake provider
    res_debug = runner.invoke(
        app,
        ["debug", str(log_file), "--provider", "fake", "--path", str(tmp_path), "--max-iters", "2"],
    )
    assert res_debug.exit_code == 0
    assert "Starting agent debugger" in res_debug.output

    # Run analyze command with fake provider
    res_analyze = runner.invoke(
        app,
        ["analyze", "project", "--provider", "fake", "--path", str(tmp_path), "--max-iters", "2"],
    )
    assert res_analyze.exit_code == 0
    assert "Starting architectural analysis" in res_analyze.output
