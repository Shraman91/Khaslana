"""Docker-based ephemeral sandbox execution for isolated testing and verification."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from khaslana.core.errors import (
    SandboxError,
    SandboxUnavailableError,
)
from khaslana.sandbox.models import ExecutionResult


def detect_image(project_dir: Path, target_file: Path | None = None) -> str:
    """Auto-detect appropriate container base image based on project structure or file type."""
    # Check target file extension first
    if target_file:
        suffix = target_file.suffix.lower()
        if suffix in {".py", ".pyw"}:
            return "python:3.11-slim"
        if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs"}:
            return "node:20-slim"
        if suffix == ".go":
            return "golang:1.22-alpine"
        if suffix == ".rs":
            return "rust:1.77-slim"
        if suffix in {".java", ".kt", ".scala"}:
            return "eclipse-temurin:17-jdk-jammy"

    # Check project directory markers
    if (project_dir / "pyproject.toml").exists() or (project_dir / "requirements.txt").exists():
        return "python:3.11-slim"
    if (project_dir / "package.json").exists():
        return "node:20-slim"
    if (project_dir / "go.mod").exists():
        return "golang:1.22-alpine"
    if (project_dir / "Cargo.toml").exists():
        return "rust:1.77-slim"
    if (project_dir / "pom.xml").exists() or (project_dir / "build.gradle").exists():
        return "eclipse-temurin:17-jdk-jammy"

    return "python:3.11-slim"


class SandboxRunner(ABC):
    """Abstract interface for sandbox execution environments."""

    @abstractmethod
    def create_temp_workspace(self, source_dir: Path) -> Path:
        """Create a writable isolated copy of the project directory."""
        ...

    @abstractmethod
    def run_command(
        self,
        cmd: str,
        workspace_dir: Path,
        image: str | None = None,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run a shell command inside an isolated container."""
        ...


class DockerRunner(SandboxRunner):
    """Ephemeral Docker container runner with strict resource limits and network isolation."""

    def __init__(
        self,
        client: Any | None = None,
        default_timeout: int = 30,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
        network_disabled: bool = True,
    ) -> None:
        self._client = client
        self.default_timeout = default_timeout
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.network_disabled = network_disabled

    def _get_client(self) -> Any:
        """Return Docker client, raising SandboxUnavailableError if unavailable."""
        if self._client is not None:
            return self._client

        try:
            import docker  # type: ignore[import-untyped]

            client = docker.from_env()
            client.ping()
            self._client = client
            return client
        except Exception as exc:
            raise SandboxUnavailableError(
                message=f"Docker is not available: {exc}",
                hint=(
                    "Please install Docker and ensure the Docker daemon / "
                    "Docker Desktop is running."
                ),
            ) from exc

    def create_temp_workspace(self, source_dir: Path) -> Path:
        """Create a writable copy of the project in a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix="khaslana_sandbox_"))

        def ignore_patterns(path: str, names: list[str]) -> set[str]:
            ignored = {
                ".git",
                "node_modules",
                "__pycache__",
                ".venv",
                "venv",
                ".mypy_cache",
                ".ruff_cache",
                ".pytest_cache",
                ".khaslana",
            }
            return {name for name in names if name in ignored or name.endswith((".pyc", ".pyo"))}

        shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True, ignore=ignore_patterns)
        return temp_dir

    def run_command(
        self,
        cmd: str,
        workspace_dir: Path,
        image: str | None = None,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command in an ephemeral container with resource constraints and cleanup."""
        client = self._get_client()
        image_to_use = image or detect_image(workspace_dir)
        actual_timeout = timeout or self.default_timeout

        volumes = {
            str(workspace_dir.resolve()): {
                "bind": "/workspace",
                "mode": "rw",
            }
        }

        container = None
        start_time = time.perf_counter()
        timed_out = False
        exit_code = -1
        stdout = ""
        stderr = ""

        try:
            container = client.containers.create(
                image=image_to_use,
                command=["sh", "-c", cmd],
                working_dir="/workspace",
                volumes=volumes,
                network_disabled=self.network_disabled,
                mem_limit=self.mem_limit,
                nano_cpus=self.nano_cpus,
                environment=environment or {},
                user="0:0",
            )
            container.start()

            # Wait for completion or timeout
            try:
                res = container.wait(timeout=actual_timeout)
                exit_code = (
                    int(res.get("StatusCode", -1))
                    if isinstance(res, dict)
                    else int(res)
                )
            except Exception as wait_exc:
                timed_out = True
                with contextlib.suppress(Exception):
                    container.kill()
                stderr = f"Execution timed out after {actual_timeout}s: {wait_exc}"

            duration = time.perf_counter() - start_time

            if not timed_out:
                raw_out = container.logs(stdout=True, stderr=False)
                raw_err = container.logs(stdout=False, stderr=True)
                stdout = (
                    raw_out.decode("utf-8", errors="replace")
                    if isinstance(raw_out, bytes)
                    else str(raw_out or "")
                )
                stderr = (
                    raw_err.decode("utf-8", errors="replace")
                    if isinstance(raw_err, bytes)
                    else str(raw_err or "")
                )

            return ExecutionResult(
                exit_code=exit_code if not timed_out else 124,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=timed_out,
                image_used=image_to_use,
            )
        except SandboxError:
            raise
        except Exception as exc:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution error: {exc}",
                duration_seconds=duration,
                timed_out=False,
                image_used=image_to_use,
            )
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.remove(force=True)


class FakeSandboxRunner(SandboxRunner):
    """Deterministic in-memory/subprocess sandbox runner for testing without Docker."""

    def __init__(self, canned_results: list[ExecutionResult] | None = None) -> None:
        self.canned_results: list[ExecutionResult] = canned_results or []
        self.invocations: list[dict[str, Any]] = []

    def create_temp_workspace(self, source_dir: Path) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="khaslana_fake_sandbox_"))
        shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True)
        return temp_dir

    def run_command(
        self,
        cmd: str,
        workspace_dir: Path,
        image: str | None = None,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
    ) -> ExecutionResult:
        self.invocations.append({
            "cmd": cmd,
            "workspace_dir": workspace_dir,
            "image": image,
            "timeout": timeout,
        })
        if self.canned_results:
            return self.canned_results.pop(0)

        # Default fallback: mock success
        return ExecutionResult(
            exit_code=0,
            stdout="Fake tests passed.",
            stderr="",
            duration_seconds=0.05,
            timed_out=False,
            image_used=image or "python:3.11-slim",
        )
