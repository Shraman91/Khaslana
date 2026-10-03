"""Core prompts and templates for Khaslana."""

SYSTEM_PROMPT = """You are Khaslana, an expert AI developer assistant designed to run local-first.
You are concise, direct, and provide high-quality code, fixes, and explanations.
When asked about code or files, cite exact file paths and line ranges where applicable.
Never guess or fabricate file contents or facts.
"""

EXPLAIN_PROMPT_TEMPLATE = """Please explain the following code/file thoroughly.
Cover:
1. High-level purpose and architectural role.
2. Key functions, classes, and exported symbols.
3. Dependencies and interactions with other parts of the project.
4. Potential edge cases, performance considerations, or failure modes.

File path: {file_path}
Language: {language}

Content:
```
{content}
```
"""

COMMIT_MSG_PROMPT = """Generate a clean, conventional commit message (e.g. feat(auth): add jwt token refresh) based on the following git diff.
Only output the commit message itself, without extra preamble or markdown quotes.

Diff:
```
{diff}
```
"""
