"""Run one provider SDK operation, printing only its JSON result on stdout."""

import contextlib
import json
import sys
from importlib import import_module


def main() -> int:
    """Run one provider request from argv and stdin; the SDK's own output goes to stderr."""
    provider, operation = sys.argv[1:]
    if provider not in {"modal", "kaggle"}:
        raise ValueError("Unknown provider.")
    request = json.loads(sys.stdin.read())
    with contextlib.redirect_stdout(sys.stderr):
        client = import_module(f"forge.providers.{provider}_api").Client()
        result = client.invoke(operation, request)
    sys.stdout.write(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
