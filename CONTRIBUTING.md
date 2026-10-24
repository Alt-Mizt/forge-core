# Contributing

Run `uv sync`, then `scripts/check.sh` before every commit. It runs ruff (format and
lint), pyright, import-linter, `scripts/lint_repo.py` and the test suite, all offline.
Keep `forge.core` free of I/O, and give every rule about money or time a test with
an answer you can check by hand.
