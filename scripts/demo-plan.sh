#!/usr/bin/env bash
# Two local accounts with 10-second sessions: print the route forge plans for the tiny job,
# then run it and show where its sessions went. Offline; spends nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
FORGE_HOME=$(mktemp -d)
export FORGE_HOME
trap 'rm -rf "$FORGE_HOME"' EXIT

uv run forge add local --name first --device cpu --session-limit 10
uv run forge add local --name second --device cpu --session-limit 10
uv run forge plan examples/tiny/train.py --hours 0.01 -- --step-delay 1
uv run forge run examples/tiny/train.py --name planned -- --step-delay 1
uv run forge status
