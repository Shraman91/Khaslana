"""ReAct agent loop with tool execution, loop detection, JSON repair, and tracing."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from khaslana.agents.models import AgentFinish, AgentStep, AgentTrace
from khaslana.agents.parser import parse_agent_output
from khaslana.core.llm import ChatMessage, LLMClient, LLMRouter
from khaslana.tools.base import ToolRegistry

REACT_SYSTEM_PROMPT = """You are an expert AI software engineering agent.
Your goal is to solve the given technical task by reasoning and using available tools.

You operate in a strict ReAct (Reason + Act) loop:
At each step, inspect previous observations and respond with a STRICT JSON object:

1. TO TAKE AN ACTION:
```json
{{
  "thought": "Your step-by-step reasoning about what you know and what to do next",
  "action": "exact_tool_name",
  "action_input": {{
    "param_name": "param_value"
  }}
}}
```

2. TO FINISH AND RETURN THE FINAL ANSWER:
```json
{{
  "thought": "Reasoning confirming that the task is resolved or sufficient data is gathered",
  "final_answer": "Complete, comprehensive explanation, fix, or resolution for the user"
}}
```

AVAILABLE TOOLS:
{tools_description}

CRITICAL RULES:
1. Output ONLY a valid JSON object matching one of the two schemas above.
2. Never repeat the exact same tool call if it already yielded an observation.
3. Synthesize your final answer once you have sufficient context to address the user request.
"""


class AgentLoop:
    """Executes ReAct agent loops with safety caps, loop detection, and rich tracing."""

    def __init__(
        self,
        llm: LLMRouter | LLMClient,
        tool_registry: ToolRegistry,
        max_iterations: int = 10,
        token_budget: int = 12000,
        console: Console | None = None,
        verbose: bool = True,
        step_callback: Callable[[AgentStep], None] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.token_budget = token_budget
        self.console = console or Console()
        self.verbose = verbose
        self.step_callback = step_callback

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimator (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def _display_step(self, step: AgentStep) -> None:
        """Render live rich panels for thought, action, and observation."""
        if not self.verbose:
            return

        panel_content = Text()
        panel_content.append(f"Step {step.step_number}\n", style="bold cyan")
        panel_content.append("💭 Thought: ", style="bold yellow")
        panel_content.append(f"{step.thought}\n\n", style="italic")
        panel_content.append("🛠️  Action: ", style="bold green")
        panel_content.append(f"{step.action}({json.dumps(step.action_input)})\n\n", style="bold")
        panel_content.append("👁️  Observation:\n", style="bold blue")

        obs_display = step.observation
        if len(obs_display) > 600:
            truncated_len = len(step.observation) - 600
            obs_display = obs_display[:600] + f"\n... [truncated {truncated_len} chars]"
        panel_content.append(obs_display, style="dim" if step.ok else "bold red")

        border_style = "green" if step.ok else "red"
        self.console.print(
            Panel(
                panel_content,
                title=f"[bold]ReAct Step {step.step_number}[/bold]",
                border_style=border_style,
                expand=False,
            )
        )

    async def run(self, task: str) -> AgentTrace:
        """Run the ReAct agent loop asynchronously until completion or max iterations."""
        start_time = time.time()
        trace = AgentTrace(task=task)

        system_msg = REACT_SYSTEM_PROMPT.format(
            tools_description=self.tools.get_descriptions()
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=f"Task: {task}"),
        ]

        action_history: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            trace.iterations = iteration

            # Check token budget; prune older observations if approaching budget
            total_estimated_tokens = sum(self._estimate_tokens(m.content) for m in messages)
            if total_estimated_tokens > self.token_budget and len(messages) > 6:
                # Keep system, initial task, and last 3 interactions
                messages = [messages[0], messages[1], *messages[-4:]]

            step_start = time.time()
            try:
                response = await self.llm.chat(messages)
                content = response.content
            except Exception as e:
                trace.success = False
                trace.error = f"LLM generation failed: {e}"
                break

            try:
                decision = parse_agent_output(content)
            except Exception as e:
                # Provide feedback to model to fix its JSON
                messages.append(ChatMessage(role="assistant", content=content))
                feedback = (
                    f"Your previous response was not valid JSON: {e}. "
                    "Please respond ONLY with a valid JSON object matching the required schema."
                )
                messages.append(ChatMessage(role="user", content=feedback))
                continue

            if isinstance(decision, AgentFinish):
                step = AgentStep(
                    step_number=iteration,
                    thought=decision.thought,
                    action="finish",
                    action_input={},
                    observation=decision.final_answer,
                    ok=True,
                    duration_seconds=round(time.time() - step_start, 3),
                )
                trace.steps.append(step)
                trace.final_answer = decision.final_answer
                trace.success = True
                if self.step_callback:
                    self.step_callback(step)
                if self.verbose:
                    self.console.print(
                        Panel(
                            f"[bold green]Final Answer:[/bold green]\n\n{decision.final_answer}",
                            title="[bold green]✓ Task Completed[/bold green]",
                            border_style="green",
                        )
                    )
                break

            # Decision is AgentAction
            action = decision.action
            action_input = decision.action_input

            # Loop detection
            action_key = f"{action}:{json.dumps(action_input, sort_keys=True)}"
            action_history.append(action_key)
            repeat_count = action_history.count(action_key)

            if repeat_count >= 4:
                obs = (
                    f"[Loop Abort] Action '{action}' with input {action_input} was executed "
                    f"{repeat_count} times without progress. Concluding loop."
                )
                step = AgentStep(
                    step_number=iteration,
                    thought=decision.thought,
                    action=action,
                    action_input=action_input,
                    observation=obs,
                    ok=False,
                    duration_seconds=round(time.time() - step_start, 3),
                )
                trace.steps.append(step)
                trace.final_answer = (
                    f"Agent loop halted: detected repeated tool calls to '{action}'. "
                    f"Last thought: {decision.thought}"
                )
                trace.success = False
                break

            # Execute tool
            tool_res = self.tools.execute(action, action_input)
            obs = tool_res.output if tool_res.ok else f"ERROR: {tool_res.error}"

            if repeat_count >= 2:
                obs = f"[Warning: Loop detected. You previously ran this exact tool call.]\n{obs}"

            step = AgentStep(
                step_number=iteration,
                thought=decision.thought,
                action=action,
                action_input=action_input,
                observation=obs,
                ok=tool_res.ok,
                duration_seconds=round(time.time() - step_start, 3),
            )
            trace.steps.append(step)
            self._display_step(step)

            if self.step_callback:
                self.step_callback(step)

            # Append to prompt history for next step
            assistant_payload = json.dumps(
                {
                    "thought": decision.thought,
                    "action": action,
                    "action_input": action_input,
                }
            )
            messages.append(ChatMessage(role="assistant", content=assistant_payload))
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Observation for {action}:\n{obs}",
                )
            )

        if not trace.final_answer and trace.success:
            trace.final_answer = (
                f"Agent reached maximum iterations ({self.max_iterations}) without finalizing. "
                "Review steps trace for partial findings."
            )
            trace.success = False

        trace.total_duration_seconds = round(time.time() - start_time, 3)
        return trace
