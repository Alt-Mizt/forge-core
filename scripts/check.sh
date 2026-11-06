#!/usr/bin/env bash
# Everything that must pass before a commit: formatting, lint, types, architecture, fast
# tests and the docs build.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run lint-imports
uv run python scripts/lint_repo.py
uv run pytest -q
uv run mkdocs build --strict --quiet
