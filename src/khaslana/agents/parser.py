"""JSON extraction, repair, and parsing for agent ReAct responses."""

from __future__ import annotations

import json
import re
from typing import Any

from khaslana.agents.models import AgentAction, AgentFinish
from khaslana.core.errors import KhaslanaError


class AgentParseError(KhaslanaError):
    """Raised when an agent response cannot be parsed as a valid action or finish."""

    pass


def repair_json_string(raw: str) -> str:
    """Attempt heuristic repairs on common LLM malformed JSON strings."""
    s = raw.strip()

    # Strip markdown code blocks if wrapped
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # Find outermost curly braces if surrounded by conversational filler
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        s = s[first_brace : last_brace + 1]

    # Replace Python literals with JSON equivalents
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)

    # Remove trailing commas before closing braces/brackets
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Normalize fancy unicode quotes
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")

    return s


def parse_agent_output(text: str) -> AgentAction | AgentFinish:
    """Parse raw LLM response into AgentAction or AgentFinish with auto-repair."""
    cleaned = repair_json_string(text)

    try:
        data: dict[str, Any] = json.loads(cleaned)
    except Exception as e:
        # Fallback: check if the model provided raw text intended as a final answer
        if "final_answer" not in text and "action" not in text:
            # If the model simply gave an answer without JSON formatting, treat as final answer
            return AgentFinish(thought="Direct final response", final_answer=text.strip())
        raise AgentParseError(
            f"Failed to parse agent JSON output: {e}",
            hint=f"Cleaned output was:\n{cleaned[:200]}...",
        ) from e

    if not isinstance(data, dict):
        raise AgentParseError(f"Agent JSON output must be a JSON object, got {type(data).__name__}")

    thought = str(data.get("thought", ""))

    # Check for final_answer
    if "final_answer" in data:
        answer = data["final_answer"]
        if not isinstance(answer, str):
            answer = json.dumps(answer, indent=2)
        return AgentFinish(thought=thought, final_answer=answer)

    # Check for action
    if "action" in data:
        action_name = str(data["action"]).strip()
        action_input = data.get("action_input", {})
        if not isinstance(action_input, dict):
            # If action_input was passed as a scalar or string, wrap in dict if possible
            action_input = {"input": action_input}
        return AgentAction(thought=thought, action=action_name, action_input=action_input)

    # If neither, but thought is present
    if thought and not data.get("action") and not data.get("final_answer"):
        return AgentFinish(thought=thought, final_answer=thought)

    raise AgentParseError(
        "Agent output JSON must contain either 'action' with 'action_input' or 'final_answer'.",
        hint=f"Keys present: {list(data.keys())}",
    )
