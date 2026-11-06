"""The run journal: everything forge knows about one run, written before each outside action.

A run lives in ``runs/<name>/`` under forge's state folder: ``journal.json``, the packed job
``job.zip``, one ``sessions/<number>/`` folder per finished session (its log, outputs and
snapshot) and, once finished, ``outputs/``. A controller that crashes loses nothing, because
the journal names each session before the provider is asked to start it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge import files

RUN_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}")
JOURNAL_VERSION = 1
# Run states. A run is "waiting" when no account had time left, "paused" when you paused it.
RUNNING, WAITING, PAUSED, FINISHED, FAILED, STOPPED = (
    "running",
    "waiting",
    "paused",
    "finished",
    "failed",
    "stopped",
)
ENDED_STATES = frozenset({FINISHED, FAILED, STOPPED})
# Session phases, in order. "recorded" means the session is closed: the usage log has its
# time, if it ran at all, and its hold is over.
STARTING, ACTIVE, RECORDED = "starting", "active", "recorded"

# A journal is a JSON object with these keys:
#   version, run, state, job {folder, script, args, requirements, digest, declaration},
#   request {seconds, gpu, memory_gb}, and sessions: a list with one entry per session:
#   number, account, gpu, handle, phase, start, seconds, hold (amount held, or null),
#   and once ended: end, outcome, and for a session that could not start, reason.
#   While a background follower carries the run: follower, its process ID.


def folder(runs: Path, run: str) -> Path:
    """The folder that holds one run."""
    if RUN_NAME.fullmatch(run) is None:
        raise ValueError(f"{run!r} is not a run name.")
    return runs / run


def journal_path(runs: Path, run: str) -> Path:
    """The journal of an existing run, or a plain refusal naming the command that lists runs."""
    path = folder(runs, run) / "journal.json"
    if not path.exists():
        raise ValueError(f"There is no run named {run}. See them with: forge status")
    return path


def name_for(script: Path, now: datetime) -> str:
    """A fresh run name from the script and the time, like ``train-20261105-081500``."""
    stem = re.sub(r"[^a-z0-9]+", "-", script.stem.lower()).strip("-") or "run"
    if not stem[0].isalpha():
        stem = f"run-{stem}"
    return f"{stem[:20]}-{now.strftime('%Y%m%d-%H%M%S')}"


def write(path: Path, journal: dict[str, Any]) -> None:
    """Replace the journal in one step, so a reader never sees it half-written."""
    encoded = json.dumps(journal, sort_keys=True, indent=1, allow_nan=False).encode()
    files.write_atomically(path, encoded)


def read(path: Path) -> dict[str, Any]:
    """Read and check a journal. A damaged one refuses, because it could hide a live session."""
    journal = json.loads(path.read_bytes())
    if not isinstance(journal, dict) or journal.get("version") != JOURNAL_VERSION:
        raise ValueError(f"The run journal {path} is damaged; restore it before continuing.")
    if not isinstance(journal.get("sessions"), list):
        raise ValueError(f"The run journal {path} has no session list.")
    return journal


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Let one forge at a time change a run, by holding the lock file beside its journal.

    A ``forge run`` in one terminal and a ``forge pause`` in another both act on the same
    run. Each takes this lock and reads the journal afresh before it changes anything.
    """
    with files.locked(path.with_name("journal.lock")):
        yield


def active(journal: dict[str, Any]) -> dict[str, Any] | None:
    """The session that has started and is not yet recorded, if there is one."""
    sessions = journal["sessions"]
    if sessions and sessions[-1]["phase"] != RECORDED:
        return sessions[-1]
    return None


def all_journals(runs: Path) -> list[dict[str, Any]]:
    """Every run's journal, oldest first."""
    paths = sorted(runs.glob("*/journal.json"), key=lambda path: path.stat().st_mtime)
    return [read(path) for path in paths]


def now_text() -> str:
    """The current UTC time in the usage log's ISO format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
