"""Self-correcting fix loop running against isolated sandbox container copies."""

from __future__ import annotations

import contextlib
import difflib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from khaslana.core.errors import KhaslanaError
from khaslana.core.llm import ChatMessage, LLMClient, LLMRouter
from khaslana.sandbox.docker_runner import SandboxRunner
from khaslana.sandbox.models import FixAttempt, FixSessionResult


def auto_detect_test_cmd(project_dir: Path, target_file: Path | None = None) -> str:
    """Infer the default test command for the workspace or target file."""
    if target_file:
        suffix = target_file.suffix.lower()
        if suffix in {".py", ".pyw"}:
            if target_file.name.startswith("test_"):
                return f"pytest {target_file.as_posix()}"
            return "pytest"
        if suffix in {".js", ".jsx", ".ts", ".tsx"}:
            return "npm test"
        if suffix == ".go":
            return "go test ./..."
        if suffix == ".rs":
            return "cargo test"

    if (project_dir / "pyproject.toml").exists() or (project_dir / "pytest.ini").exists():
        return "pytest"
    if (project_dir / "package.json").exists():
        return "npm test"
    if (project_dir / "go.mod").exists():
        return "go test ./..."
    if (project_dir / "Cargo.toml").exists():
        return "cargo test"

    return "pytest"


def extract_code_fix(raw_response: str) -> tuple[str, str]:
    """Extract proposed replacement code and optional explanation from LLM output."""
    raw = raw_response.strip()

    # Try JSON format first
    if "{" in raw and "}" in raw:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            if isinstance(data, dict):
                code = str(data.get("fixed_code", "")).strip()
                explanation = str(data.get("explanation", "")).strip()
                if code:
                    return code, explanation
        except Exception:
            pass

    # Try code block extraction (```lang\n...``` or ```\n...```)
    pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    matches: list[str] = pattern.findall(raw)
    if matches:
        # Take the longest code block
        longest = max(matches, key=len).strip()
        explanation = raw.split("```")[0].strip()
        return longest, explanation

    return raw, ""


def compute_unified_diff(original: str, modified: str, file_name: str) -> str:
    """Generate unified diff string between original and modified file contents."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{file_name}",
        tofile=f"b/{file_name}",
    )
    return "".join(diff)


class SelfCorrectingLoop:
    """Orchestrates iterative fix suggestions and sandbox test verification."""

    def __init__(
        self,
        runner: SandboxRunner,
        llm: LLMRouter | LLMClient,
        project_root: Path,
        file_rel_path: str | Path,
        test_cmd: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.runner = runner
        self.llm = llm
        self.project_root = Path(project_root).resolve()
        self.file_rel_path = Path(file_rel_path)
        self.max_attempts = max_attempts

        self.target_file = (self.project_root / self.file_rel_path).resolve()
        if not self.target_file.exists():
            raise KhaslanaError(
                message=(
                    f"Target file '{self.file_rel_path}' does not exist in '{self.project_root}'."
                ),
                hint=(
                    "Provide a valid relative or absolute path to an existing file in the project."
                ),
            )

        self.test_cmd = test_cmd or auto_detect_test_cmd(self.project_root, self.target_file)

    async def run(self) -> FixSessionResult:
        """Run the self-correcting fix loop inside an isolated sandbox copy."""
        original_code = self.target_file.read_text(encoding="utf-8")
        current_code = original_code
        file_name_str = self.file_rel_path.as_posix()

        # Create temporary sandbox workspace (never touches real working tree)
        temp_ws = self.runner.create_temp_workspace(self.project_root)

        attempts: list[FixAttempt] = []
        try:
            # Baseline execution: check if tests currently pass
            baseline_res = self.runner.run_command(self.test_cmd, workspace_dir=temp_ws)
            if baseline_res.is_success:
                return FixSessionResult(
                    file_path=file_name_str,
                    test_cmd=self.test_cmd,
                    success=True,
                    attempts=[],
                    final_code=original_code,
                    final_diff=None,
                    applied=False,
                )

            last_result = baseline_res

            for attempt_num in range(1, self.max_attempts + 1):
                prompt = self._build_fix_prompt(
                    file_name=file_name_str,
                    current_code=current_code,
                    test_cmd=self.test_cmd,
                    last_result=last_result,
                    attempt_num=attempt_num,
                )

                resp = await self.llm.chat([ChatMessage(role="user", content=prompt)])
                response = resp.content
                proposed_code, explanation = extract_code_fix(response)

                if not proposed_code.strip():
                    proposed_code = current_code

                diff = compute_unified_diff(original_code, proposed_code, file_name_str)

                # Write fix ONLY to sandbox temp workspace copy
                sandbox_file = temp_ws / self.file_rel_path
                sandbox_file.parent.mkdir(parents=True, exist_ok=True)
                sandbox_file.write_text(proposed_code, encoding="utf-8")

                # Re-run tests in sandbox
                test_res = self.runner.run_command(self.test_cmd, workspace_dir=temp_ws)

                attempt = FixAttempt(
                    attempt_number=attempt_num,
                    proposed_code=proposed_code,
                    diff=diff,
                    result=test_res,
                    success=test_res.is_success,
                    explanation=explanation,
                )
                attempts.append(attempt)

                if test_res.is_success:
                    return FixSessionResult(
                        file_path=file_name_str,
                        test_cmd=self.test_cmd,
                        success=True,
                        attempts=attempts,
                        final_code=proposed_code,
                        final_diff=diff,
                        applied=False,
                    )

                # Update context for next attempt
                current_code = proposed_code
                last_result = test_res

            return FixSessionResult(
                file_path=file_name_str,
                test_cmd=self.test_cmd,
                success=False,
                attempts=attempts,
                final_code=None,
                final_diff=None,
                applied=False,
            )
        finally:
            with contextlib.suppress(Exception):
                shutil.rmtree(temp_ws, ignore_errors=True)

    def _build_fix_prompt(
        self,
        file_name: str,
        current_code: str,
        test_cmd: str,
        last_result: Any,
        attempt_num: int,
    ) -> str:
        return f"""You are a senior software engineer fixing a bug or failing test.
Target file to fix: {file_name}
Attempt: {attempt_num} of {self.max_attempts}
Test command: {test_cmd}

=== Current Content of {file_name} ===
{current_code}

=== Test Failure Output (Exit Code: {last_result.exit_code}) ===
STDOUT:
{last_result.stdout[:2000]}
STDERR:
{last_result.stderr[:2000]}

Instructions:
1. Diagnose the root cause of the test failure.
2. Provide the complete, updated content of {file_name}.
3. Wrap your code inside a single markdown code block (e.g. ```python ... ```).
Do not truncate or omit lines. Return the entire corrected file content.
"""
