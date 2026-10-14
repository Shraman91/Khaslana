"""Code review engine, diff explainer, and conventional commit generator."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from khaslana.agents.parser import repair_json_string
from khaslana.core.llm import ChatMessage, LLMRouter
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.symbols import SymbolTable
from khaslana.tools.git import GitInspector, ReviewFinding, ReviewReport

REVIEW_SYSTEM_PROMPT = """You are a senior principal staff engineer reviewing code changes.
Perform a thorough, rigorous, and constructive code review.
Focus on:
1. Correctness & Bugs (edge cases, off-by-one, null/None handling, race conditions)
2. Security (injection, authentication, authorization, leaked secrets, unsafe deserialization)
3. Performance (algorithmic complexity, redundant queries, memory leaks, blocking I/O)
4. Code Style & Architecture (maintainability, separation of concerns, API consistency)

CRITICAL: Respond ONLY with a valid JSON object matching this schema:
{
  "summary": "Concise 1-3 sentence summary of the diff quality and overall readiness",
  "findings": [
    {
      "file": "file_path.py",
      "line": 42,
      "severity": "bug",  // must be one of: "bug", "security", "perf", "style"
      "title": "Short title describing the issue",
      "description": "Clear explanation of the problem, potential impact, and recommendation",
      "suggested_patch": "Optional code snippet or diff demonstrating the fix"
    }
  ]
}

If no issues are found, return an empty findings list `[]` with a complimentary summary.
"""

COMMIT_MSG_SYSTEM_PROMPT = """You are an expert Git practitioner.
Analyze the following git staged diff and produce a high quality Conventional Commit message.

Format Requirements:
1. Header: <type>(<scope>): <imperative summary in lowercase, max 72 chars, no trailing period>
   - types: feat, fix, refactor, perf, test, docs, chore, style
2. Blank line
3. Body: A bulleted list explaining what was changed and why.

Do not include markdown code block formatting or filler. Output ONLY the commit message.
"""

EXPLAIN_DIFF_SYSTEM_PROMPT = """You are an expert software engineer.
Explain the following git diff in clear, structured technical language:
1. What changes are being introduced across each modified file.
2. The architectural rationale and purpose behind these modifications.
3. Impact on callers, callees, or system behavior.
4. Any potential risks or notable testing considerations.
"""


class CodeReviewer:
    """Executes diff reviews, diff explanations, and commit message generation."""

    def __init__(
        self,
        router: LLMRouter,
        root_dir: Path | str = ".",
        console: Console | None = None,
    ) -> None:
        self.router = router
        self.root_dir = Path(root_dir).resolve()
        self.console = console or Console()
        self.git = GitInspector(root_dir=self.root_dir)

    def _get_symbols_and_graph(self) -> tuple[SymbolTable | None, CallGraph | None]:
        sym_db = self.root_dir / ".khaslana" / "symbols.db"
        if sym_db.is_file():
            st = SymbolTable(sym_db)
            cg = CallGraph(st)
            return st, cg
        return None, None

    async def review(
        self, staged: bool = False, branch: str | None = None
    ) -> ReviewReport:
        """Run a semantic, diff-aware code review against staged changes or a branch."""
        st, cg = self._get_symbols_and_graph()
        enriched = self.git.get_enriched_context(
            staged=staged,
            branch=branch,
            symbol_table=st,
            call_graph=cg,
        )

        target_desc = (
            f"branch '{branch}'" if branch else ("staged changes" if staged else "working tree")
        )

        if not enriched.diffs or not enriched.raw_diff.strip():
            return ReviewReport(
                target=target_desc,
                findings=[],
                summary=f"No diff found for {target_desc}.",
                total_files_analyzed=0,
            )

        # Assemble prompt with enriched context
        context_parts = [f"DIFF FOR {target_desc.upper()}:\n```diff\n{enriched.raw_diff}\n```\n"]

        if enriched.touched_symbols:
            context_parts.append("TOUCHED SYMBOLS & CALL GRAPH CONTEXT:")
            for sym in enriched.touched_symbols:
                callers = enriched.callers.get(sym.name, [])
                callees = enriched.callees.get(sym.name, [])
                c_str = f", Callers: {callers}" if callers else ""
                ce_str = f", Callees: {callees}" if callees else ""
                loc = f"{sym.file}:{sym.line_range[0]}-{sym.line_range[1]}"
                entry = f"• {sym.symbol_type.upper()} {sym.name} ({loc}){c_str}{ce_str}"
                context_parts.append(entry)

        user_content = "\n".join(context_parts)
        messages = [
            ChatMessage(role="system", content=REVIEW_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]

        response = await self.router.chat(messages)
        cleaned = repair_json_string(response.content)

        try:
            data = json.loads(cleaned)
            findings_data = data.get("findings", [])
            findings = [ReviewFinding.model_validate(f) for f in findings_data]
            summary = str(data.get("summary", "Review complete."))
        except Exception:
            findings = []
            summary = response.content.strip()

        return ReviewReport(
            target=target_desc,
            findings=findings,
            summary=summary,
            total_files_analyzed=len(enriched.diffs),
        )

    async def explain_diff(
        self, staged: bool = False, branch: str | None = None
    ) -> str:
        """Provide an architectural explanation of the diff."""
        st, cg = self._get_symbols_and_graph()
        enriched = self.git.get_enriched_context(
            staged=staged,
            branch=branch,
            symbol_table=st,
            call_graph=cg,
        )

        target_desc = (
            f"branch '{branch}'" if branch else ("staged changes" if staged else "working tree")
        )

        if not enriched.diffs or not enriched.raw_diff.strip():
            return f"No diff detected for {target_desc}."

        context_parts = [
            f"DIFF FOR {target_desc.upper()}:\n```diff\n{enriched.raw_diff}\n```\n"
        ]
        if enriched.touched_symbols:
            context_parts.append("MODIFIED SYMBOLS CONTEXT:")
            for sym in enriched.touched_symbols:
                callers = enriched.callers.get(sym.name, [])
                callees = enriched.callees.get(sym.name, [])
                callers_str = callers or "None"
                callees_str = callees or "None"
                entry = (
                    f"- {sym.name} in {sym.file} "
                    f"(Callers: {callers_str}, Callees: {callees_str})"
                )
                context_parts.append(entry)

        messages = [
            ChatMessage(role="system", content=EXPLAIN_DIFF_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(context_parts)),
        ]
        res = await self.router.chat(messages)
        return res.content.strip()

    async def generate_commit_message(self) -> str:
        """Generate a conventional commit message from staged changes."""
        diff_text = self.git.get_diff_text(staged=True)
        if not diff_text.strip():
            return ""

        messages = [
            ChatMessage(role="system", content=COMMIT_MSG_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Staged Diff:\n```diff\n{diff_text}\n```"),
        ]
        res = await self.router.chat(messages)
        return res.content.strip()


def format_review_rich(report: ReviewReport, console: Console) -> None:
    """Format and print review findings as a styled Rich table and panels."""
    console.print(f"\n[bold blue]Code Review Report for {report.target}[/bold blue]")
    console.print(f"[dim]Analyzed {report.total_files_analyzed} file(s)[/dim]\n")

    if not report.findings:
        console.print(
            Panel(
                f"[bold green]✓ No issues found.[/bold green]\n\n{report.summary}",
                border_style="green",
            )
        )
        return

    table = Table(title=f"Review Findings ({len(report.findings)})", expand=True)
    table.add_column("Severity", justify="center", width=10)
    table.add_column("File:Line", style="bold green", width=25)
    table.add_column("Title", style="bold", width=35)
    table.add_column("Description", style="white")

    severity_styles = {
        "bug": "bold red",
        "security": "bold magenta",
        "perf": "bold yellow",
        "style": "bold cyan",
    }

    for f in report.findings:
        sev_style = severity_styles.get(f.severity, "white")
        loc = f"{f.file}:{f.line}" if f.line is not None else f.file
        table.add_row(
            f"[{sev_style}]{f.severity.upper()}[/{sev_style}]",
            loc,
            f.title,
            f.description,
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {report.summary}\n")

    # Show suggested patches if any
    for i, f in enumerate(report.findings, start=1):
        if f.suggested_patch:
            loc = f"{f.file}:{f.line}" if f.line is not None else f.file
            console.print(
                Panel(
                    f.suggested_patch,
                    title=f"[cyan]Suggested Patch #{i} ({loc})[/cyan]",
                    border_style="dim",
                )
            )


def format_review_markdown(report: ReviewReport) -> str:
    """Format review report as GitHub-flavored Markdown for CI."""
    lines = [
        f"# Code Review Report ({report.target})",
        "",
        f"**Summary:** {report.summary}",
        f"**Files Analyzed:** {report.total_files_analyzed}",
        f"**Total Findings:** {len(report.findings)}",
        "",
    ]

    if not report.findings:
        lines.append("🎉 **No issues detected!**")
        return "\n".join(lines)

    lines.append("| Severity | File:Line | Title | Description |")
    lines.append("| :--- | :--- | :--- | :--- |")

    for f in report.findings:
        loc = f"`{f.file}:{f.line}`" if f.line is not None else f"`{f.file}`"
        lines.append(f"| **{f.severity.upper()}** | {loc} | {f.title} | {f.description} |")

    lines.append("")
    for i, f in enumerate(report.findings, start=1):
        if f.suggested_patch:
            loc = f"{f.file}:{f.line}" if f.line is not None else f.file
            lines.append(f"### Suggested Patch #{i} ({loc})")
            lines.append("```diff")
            lines.append(f.suggested_patch)
            lines.append("```\n")

    return "\n".join(lines)


def format_review_json(report: ReviewReport) -> str:
    """Format review report as JSON string for automated tooling."""
    return report.model_dump_json(indent=2)
