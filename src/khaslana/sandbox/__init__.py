"""Sandbox execution module for Khaslana."""

from khaslana.sandbox.docker_runner import (
    DockerRunner,
    FakeSandboxRunner,
    SandboxRunner,
    detect_image,
)
from khaslana.sandbox.fix_loop import (
    SelfCorrectingLoop,
    auto_detect_test_cmd,
    compute_unified_diff,
    extract_code_fix,
)
from khaslana.sandbox.models import ExecutionResult, FixAttempt, FixSessionResult

__all__ = [
    "DockerRunner",
    "ExecutionResult",
    "FakeSandboxRunner",
    "FixAttempt",
    "FixSessionResult",
    "SandboxRunner",
    "SelfCorrectingLoop",
    "auto_detect_test_cmd",
    "compute_unified_diff",
    "detect_image",
    "extract_code_fix",
]
