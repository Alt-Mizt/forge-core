#!/usr/bin/env bash
# Exercise the CLI with a script that has no forge imports.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/resume_proof.py
