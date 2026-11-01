#!/usr/bin/env bash
# One run, two local accounts with 10-second sessions: the job hops at every session limit
# and must end with exactly the weights of an uninterrupted run. Offline; spends nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
FORGE_HOME=$(mktemp -d)
export FORGE_HOME
work=$(mktemp -d)
trap 'rm -rf "$FORGE_HOME" "$work"' EXIT

uv run forge add local --name first --device cpu --session-limit 10
uv run forge add local --name second --device cpu --session-limit 10
uv run forge accounts
uv run forge run examples/tiny/train.py --name hop -- --step-delay 1
uv run forge status

moved=$(cat "$FORGE_HOME/state/runs/hop/outputs/weights-sha256.txt")
straight=$(cd "$work" && uv run --project "$OLDPWD" python "$OLDPWD/examples/tiny/train.py")
echo "hopped        $moved"
echo "uninterrupted $straight"
test "$moved" = "$straight"
echo "Same weights after the hops."
