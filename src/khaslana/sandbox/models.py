"""Pydantic models for sandbox execution and self-correcting fix loop."""

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Result of running a command inside the sandbox."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    image_used: str = ""

    @property
    def is_success(self) -> bool:
        """True if execution completed with exit code 0 and did not time out."""
        return self.exit_code == 0 and not self.timed_out


class FixAttempt(BaseModel):
    """A single attempt in the self-correcting fix loop."""

    attempt_number: int
    proposed_code: str
    diff: str
    result: ExecutionResult
    success: bool
    explanation: str = ""


class FixSessionResult(BaseModel):
    """Aggregate result of a self-correcting fix session."""

    file_path: str
    test_cmd: str
    success: bool
    attempts: list[FixAttempt] = Field(default_factory=list)
    final_code: str | None = None
    final_diff: str | None = None
    applied: bool = False
