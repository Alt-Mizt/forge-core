#!/usr/bin/env bash
# Sequential baselines, pauses and fresh continuations in both directions.
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
uv run python examples/train_migrate.py --device cpu --output "$work/cpu"
uv run python examples/train_migrate.py --device cpu --stop-at 6 --snapshot "$work/paused" --output "$work/unused"
uv run python examples/train_migrate.py --device mps --resume "$work/paused" --output "$work/moved"
uv run python scripts/compare_migration.py "$work/cpu" "$work/moved"
uv run python examples/train_migrate.py --device mps --output "$work/mps"
uv run python examples/train_migrate.py --device mps --stop-at 6 --snapshot "$work/reverse-paused" --output "$work/unused"
uv run python examples/train_migrate.py --device cpu --resume "$work/reverse-paused" --output "$work/reverse-moved"
uv run python scripts/compare_migration.py "$work/mps" "$work/reverse-moved" --source mps
