# Contributing to Khaslana

Thank you for your interest in contributing to Khaslana! We welcome issues, bug reports, feature requests, and pull requests.

## Development Setup

Khaslana requires Python 3.11 or higher.

### 1. Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/Shraman91/Khaslana.git
cd khaslana

# Create and activate virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\Activate.ps1
```

### 2. Install Development Dependencies

Install Khaslana in editable mode with all extras and developer tools:

```bash
python -m pip install --upgrade pip
pip install -e ".[all]"
pip install ruff mypy pytest pytest-cov pytest-asyncio respx syrupy textual-snapshot
```

---

## Coding Standards & Quality Verification

Before submitting pull requests, ensure all checks pass:

### 1. Linting & Formatting

```bash
# Check code style and rules
ruff check src tests benchmarks

# Check code formatting
ruff format --check src tests benchmarks
```

### 2. Static Type Checking

Khaslana strictly requires type hints across all source code:

```bash
mypy src
```

### 3. Test Suite & Coverage

Run the unit test suite:

```bash
# Run tests excluding daemons and external network
pytest -m "not docker and not ollama and not network" --cov=khaslana
```

---

## Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(...)`: New features
- `fix(...)`: Bug fixes
- `test(...)`: Adding or updating tests
- `docs(...)`: Documentation changes
- `refactor(...)`: Code refactoring without behavioral change
- `chore(...)`: Maintenance or tooling updates

---

## Reporting Security Issues

Please review [`SECURITY.md`](SECURITY.md) for vulnerability reporting guidelines.
