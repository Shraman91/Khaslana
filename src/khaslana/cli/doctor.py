"""Doctor diagnostics command."""

from __future__ import annotations

import os
import shutil
import subprocess

import httpx
from rich.console import Console
from rich.table import Table

from khaslana.core.config import KhaslanaConfig


async def check_ollama(url: str) -> tuple[bool, str]:
    """Check Ollama service status and downloaded models."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{url}/api/tags")
            if res.status_code == 200:
                models = res.json().get("models", [])
                if not models:
                    return True, "Running (0 models found - run `ollama pull llama3`)"
                names = [m.get("name", "") for m in models[:3]]
                suffix = f", +{len(models)-3} more" if len(models) > 3 else ""
                return True, f"Running ({', '.join(names)}{suffix})"
            return False, f"HTTP status {res.status_code}"
    except Exception:
        return False, f"Not running or unreachable at {url}"


def check_binary_version(name: str, args: list[str]) -> tuple[bool, str]:
    """Check if a CLI binary is in PATH and runnable."""
    path = shutil.which(name)
    if not path:
        return False, "Not found in PATH"
    try:
        res = subprocess.run(
            [name, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = res.stdout or res.stderr
        first_line = output.strip().splitlines()[0] if output else "Installed"
        return True, first_line[:50]
    except Exception:
        return True, f"Installed at {path}"


def check_docker() -> tuple[bool, str]:
    """Check if Docker CLI exists and docker daemon is responsive."""
    if not shutil.which("docker"):
        return False, "Docker CLI not found in PATH"
    try:
        res = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return True, "Daemon running & responsive"
        return False, "CLI installed, but daemon not running"
    except Exception:
        return False, "Docker check failed"


async def run_doctor(config: KhaslanaConfig, console: Console) -> bool:
    """Run full diagnostic checks and print report."""
    table = Table(title="Khaslana System Doctor", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=18)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Details", style="white")

    all_ok = True

    # 1. Ollama check
    ollama_ok, ollama_det = await check_ollama(config.llm.ollama_url)
    table.add_row("Ollama (Local)", "✓" if ollama_ok else "!", ollama_det)

    # 2. Git check
    git_ok, git_det = check_binary_version("git", ["--version"])
    table.add_row("Git", "✓" if git_ok else "✗", git_det)
    if not git_ok:
        all_ok = False

    # 3. Docker check
    docker_ok, docker_det = check_docker()
    table.add_row("Docker", "✓" if docker_ok else "!", docker_det)

    # 4. Kubectl check
    k8s_ok, k8s_det = check_binary_version("kubectl", ["version", "--client"])
    table.add_row(
        "Kubectl",
        "✓" if k8s_ok else "-",
        k8s_det if k8s_ok else "Optional (for k8s deploy doctor)",
    )

    # 5. Cloud API keys check
    openai_present = bool(config.llm.openai_api_key or os.getenv("OPENAI_API_KEY"))
    gemini_present = bool(config.llm.gemini_api_key or os.getenv("GEMINI_API_KEY"))
    table.add_row(
        "OpenAI Fallback",
        "✓" if openai_present else "-",
        "Configured" if openai_present else "Not set (optional fallback)",
    )
    table.add_row(
        "Gemini Fallback",
        "✓" if gemini_present else "-",
        "Configured" if gemini_present else "Not set (optional fallback)",
    )

    console.print(table)
    return all_ok
