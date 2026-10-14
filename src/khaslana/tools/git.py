"""Git repository operations, diff extraction, and symbol enrichment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from pydantic import BaseModel, Field

from khaslana.core.errors import KhaslanaError
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.models import Symbol
from khaslana.intelligence.symbols import SymbolTable


class GitError(KhaslanaError):
    """Raised on git operation failures."""

    pass


class GitDiff(BaseModel):
    """Structured git diff information for a file."""

    file: str
    change_type: str  # 'A', 'M', 'D', 'R'
    old_path: str | None = None
    new_path: str | None = None
    patch: str = ""
    added_lines: list[tuple[int, str]] = Field(default_factory=list)
    deleted_lines: list[tuple[int, str]] = Field(default_factory=list)


class EnrichedDiffContext(BaseModel):
    """Diff context enriched with AST symbols, callers, and callees."""

    diffs: list[GitDiff]
    touched_symbols: list[Symbol] = Field(default_factory=list)
    callers: dict[str, list[str]] = Field(default_factory=dict)
    callees: dict[str, list[str]] = Field(default_factory=dict)
    raw_diff: str = ""


class ReviewFinding(BaseModel):
    """An individual finding or issue detected during code review."""

    file: str
    line: int | None = None
    severity: Literal["bug", "security", "perf", "style"]
    title: str
    description: str
    suggested_patch: str | None = None


class ReviewReport(BaseModel):
    """Complete code review report with findings and summary."""

    target: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = ""
    total_files_analyzed: int = 0


class GitInspector:
    """Inspects git repository status, diffs, blame, and history."""

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()
        self._repo: git.Repo | None = None

    def get_repo(self) -> git.Repo:
        """Obtain git.Repo instance with graceful error handling."""
        if self._repo is not None:
            return self._repo

        try:
            self._repo = git.Repo(str(self.root_dir), search_parent_directories=True)
            return self._repo
        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            raise GitError(
                f"Directory '{self.root_dir}' is not a git repository.",
                hint="Run `git init` to initialize a git repository.",
            ) from e
        except Exception as e:
            raise GitError(f"Failed to access git repository: {e}") from e

    def get_diff_text(self, staged: bool = False, branch: str | None = None) -> str:
        """Fetch raw unified diff text."""
        repo = self.get_repo()
        try:
            if branch:
                # Diff between branch and HEAD / working tree
                return str(repo.git.diff(branch))
            if staged:
                # Staged diff against HEAD (or empty tree if initial commit)
                try:
                    return str(repo.git.diff("--staged"))
                except Exception:
                    return str(repo.git.diff("--cached"))
            # Unstaged working tree diff
            return str(repo.git.diff())
        except Exception as e:
            raise GitError(f"Failed to retrieve git diff: {e}") from e

    def get_changed_files(self, staged: bool = False, branch: str | None = None) -> list[str]:
        """List relative paths of changed files."""
        repo = self.get_repo()
        try:
            if branch:
                diff_index = repo.commit(branch).diff(None)
            elif staged:
                try:
                    diff_index = repo.head.commit.diff()
                except Exception:
                    diff_index = repo.index.diff(None)
            else:
                diff_index = repo.index.diff(None)

            changed: list[str] = []
            for item in diff_index:
                if item.a_path:
                    changed.append(item.a_path)
                elif item.b_path:
                    changed.append(item.b_path)
            return sorted(list(set(changed)))
        except Exception:
            # Fallback parsing raw diff text
            diff_text = self.get_diff_text(staged=staged, branch=branch)
            matches = re.findall(r"^diff --git a/(.*?) b/(.*?)$", diff_text, re.MULTILINE)
            files = [b for _, b in matches]
            return sorted(list(set(files)))

    def get_structured_diff(
        self, staged: bool = False, branch: str | None = None
    ) -> list[GitDiff]:
        """Parse git diff into structured GitDiff objects with added/deleted line numbers."""
        diff_text = self.get_diff_text(staged=staged, branch=branch)
        if not diff_text.strip():
            return []

        file_diffs: list[GitDiff] = []
        raw_chunks = re.split(r"(?=diff --git )", diff_text)

        for chunk in raw_chunks:
            if not chunk.strip():
                continue

            match = re.search(r"diff --git a/(.*?) b/(.*?)\n", chunk)
            if not match:
                continue

            old_file = match.group(1)
            new_file = match.group(2)
            file_name = new_file if new_file != "/dev/null" else old_file

            change_type = "M"
            if "new file mode" in chunk:
                change_type = "A"
            elif "deleted file mode" in chunk:
                change_type = "D"
            elif "rename from" in chunk:
                change_type = "R"

            added_lines: list[tuple[int, str]] = []
            deleted_lines: list[tuple[int, str]] = []

            # Parse unified diff hunks
            hunk_regex = re.compile(
                r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
            )
            hunk_parts = hunk_regex.split(chunk)

            if len(hunk_parts) > 1:
                idx = 1
                while idx < len(hunk_parts):
                    new_start = int(hunk_parts[idx + 1])
                    hunk_body = hunk_parts[idx + 2]
                    idx += 3

                    curr_new_line = new_start
                    for line in hunk_body.splitlines():
                        if line.startswith("+") and not line.startswith("+++"):
                            added_lines.append((curr_new_line, line[1:]))
                            curr_new_line += 1
                        elif line.startswith("-") and not line.startswith("---"):
                            deleted_lines.append((curr_new_line, line[1:]))
                        else:
                            curr_new_line += 1

            file_diffs.append(
                GitDiff(
                    file=file_name,
                    change_type=change_type,
                    old_path=old_file,
                    new_path=new_file,
                    patch=chunk,
                    added_lines=added_lines,
                    deleted_lines=deleted_lines,
                )
            )

        return file_diffs

    def get_enriched_context(
        self,
        staged: bool = False,
        branch: str | None = None,
        symbol_table: SymbolTable | None = None,
        call_graph: CallGraph | None = None,
    ) -> EnrichedDiffContext:
        """Enrich diffs with symbol definitions and caller/callee context."""
        diffs = self.get_structured_diff(staged=staged, branch=branch)
        raw_diff = self.get_diff_text(staged=staged, branch=branch)

        touched_symbols: list[Symbol] = []
        callers_map: dict[str, list[str]] = {}
        callees_map: dict[str, list[str]] = {}

        if not symbol_table or not diffs:
            return EnrichedDiffContext(diffs=diffs, raw_diff=raw_diff)

        cg = call_graph or CallGraph(symbol_table)

        for d in diffs:
            file_symbols = symbol_table.get_symbols_in_file(d.file)
            changed_line_numbers = {line_no for line_no, _ in d.added_lines}

            for sym in file_symbols:
                s_start, s_end = sym.line_range
                # If any added line falls within symbol range
                if any(s_start <= ln <= s_end for ln in changed_line_numbers):
                    touched_symbols.append(sym)
                    callers = cg.get_callers(sym.name, max_depth=1)
                    callees = cg.get_callees(sym.name, max_depth=1)
                    if callers:
                        callers_map[sym.name] = callers
                    if callees:
                        callees_map[sym.name] = callees

        return EnrichedDiffContext(
            diffs=diffs,
            touched_symbols=touched_symbols,
            callers=callers_map,
            callees=callees_map,
            raw_diff=raw_diff,
        )

    def get_log(self, max_count: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent commit log history."""
        repo = self.get_repo()
        try:
            commits = list(repo.iter_commits(max_count=max_count))
            return [
                {
                    "hexsha": c.hexsha[:8],
                    "author": f"{c.author.name} <{c.author.email}>",
                    "date": c.committed_datetime.isoformat(),
                    "summary": c.summary,
                    "message": c.message.strip(),
                }
                for c in commits
            ]
        except Exception as e:
            raise GitError(f"Failed to read git commit log: {e}") from e

    def get_blame(self, file_path: str) -> list[dict[str, Any]]:
        """Retrieve git blame information for a file."""
        repo = self.get_repo()
        try:
            blame_data: Any = repo.blame("HEAD", file_path)
            if not blame_data:
                return []
            results = []
            for item in blame_data:
                commit = item[0]
                lines = item[1]
                author_name = getattr(commit.author, "name", str(commit.author))
                dt = getattr(commit, "committed_datetime", None)
                date_str = dt.isoformat() if dt else ""
                results.append(
                    {
                        "commit": str(getattr(commit, "hexsha", ""))[:8],
                        "author": author_name,
                        "date": date_str,
                        "lines": lines,
                    }
                )
            return results
        except Exception as e:
            raise GitError(f"Failed to get blame for {file_path}: {e}") from e
