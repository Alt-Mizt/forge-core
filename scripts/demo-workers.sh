#!/usr/bin/env bash
# Four data workers before SIGTERM, two after fresh-process resume.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/workers_proof.py
