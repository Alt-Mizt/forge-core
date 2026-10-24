#!/usr/bin/env bash
# Everything that must pass before a commit: formatting, lint, types, architecture and fast tests.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run lint-imports
uv run python scripts/lint_repo.py
uv run pytest -q
