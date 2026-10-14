"""Pydantic models for the Khaslana agent system."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """An action step decided by the agent planner."""

    thought: str = Field(
        default="", description="Reasoning and thought process leading to this action"
    )
    action: str = Field(..., description="Name of the tool to execute")
    action_input: dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass to the tool"
    )


class AgentFinish(BaseModel):
    """Final answer produced by the agent upon task completion."""

    thought: str = Field(default="", description="Reasoning confirming the task is complete")
    final_answer: str = Field(..., description="Comprehensive final answer or resolution")


class AgentStep(BaseModel):
    """A completed execution step containing action and resulting observation."""

    step_number: int
    thought: str
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str
    ok: bool = True
    timestamp: float = Field(default_factory=time.time)
    duration_seconds: float = 0.0


class AgentTrace(BaseModel):
    """Full execution trace of an agent run."""

    task: str
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str | None = None
    iterations: int = 0
    total_duration_seconds: float = 0.0
    success: bool = True
    error: str | None = None
