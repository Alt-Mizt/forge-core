#!/usr/bin/env bash
# Compare fresh-process resumes at ordinary and epoch-boundary steps.
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
straight=$(uv run python examples/train_tiny.py)
for stop in 1 6 13; do
    uv run python examples/train_tiny.py --stop-at "$stop" --snapshot "$work/stop-$stop"
    resumed=$(uv run python examples/train_tiny.py --resume "$work/stop-$stop")
    printf 'stop=%s uninterrupted %s\n' "$stop" "$straight"
    printf 'stop=%s resumed       %s\n' "$stop" "$resumed"
    test "$straight" = "$resumed"
done
