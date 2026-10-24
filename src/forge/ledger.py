"""The usage log: one JSON line per recorded session, each hashing the line before it.

Amounts are written as decimal text so they stay exact. Reading checks every line against
its predecessor, so an edited or truncated line is noticed instead of turning into extra
free time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from forge import files
from forge.core.quota import Unit

FIELDS = frozenset(
    {
        "session_id",
        "account",
        "gpu",
        "gpu_count",
        "start",
        "end",
        "unit",
        "amount",
        "run",
        "job",
        "job_digest",
        "prev_sha256",
    }
)
TEXT_FIELDS = ("session_id", "account", "gpu", "run", "job", "job_digest")


@dataclass(frozen=True)
class Entry:
    """The fields of one recorded session that the time-left calculation needs."""

    session_id: str
    account: str
    start: datetime
    end: datetime
    unit: Unit
    amount: Decimal


def read(path: Path) -> list[Entry]:
    """Check every line of the usage log and return its sessions, oldest first."""
    previous: str | None = None
    entries: list[Entry] = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError(f"The usage log {path} ends with a half-written line.")
        data = json.loads(line)
        if data.get("prev_sha256") != previous:
            raise ValueError(f"The usage log {path} was edited: its hash chain is broken.")
        entry = _entry(data)
        if any(entry.session_id == known.session_id for known in entries):
            raise ValueError(f"The usage log records session {entry.session_id} twice.")
        entries.append(entry)
        previous = hashlib.sha256(line).hexdigest()
    return entries


def append(path: Path, session: dict[str, Any]) -> None:
    """Add one session, given without ``prev_sha256``, under the log's lock."""
    with files.locked(path.with_name(path.name + ".lock")):
        if any(session["session_id"] == entry.session_id for entry in read(path)):
            raise ValueError(f"Session {session['session_id']} is already in the usage log.")
        lines = path.read_bytes().splitlines(keepends=True)
        previous = hashlib.sha256(lines[-1]).hexdigest() if lines else None
        document = {**session, "prev_sha256": previous}
        _entry(document)
        line = json.dumps(document, sort_keys=True, allow_nan=False) + "\n"
        files.append_synced(path, line.encode())


def timestamp(value: str) -> datetime:
    """Parse an ISO timestamp into UTC."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.utcoffset() is None:
        raise ValueError(f"The time {value} has no time zone.")
    return moment.astimezone(UTC)


def _entry(data: dict[str, Any]) -> Entry:
    """Check one line's fields and return the entry they describe."""
    if set(data) != FIELDS:
        raise ValueError("A usage log line has missing or unknown fields.")
    if any(not isinstance(data[field], str) or not data[field] for field in TEXT_FIELDS):
        raise ValueError("A usage log line has an empty name field.")
    if type(data["gpu_count"]) is not int or data["gpu_count"] < 1:
        raise ValueError("A usage log line has an invalid GPU count.")
    start, end = timestamp(data["start"]), timestamp(data["end"])
    if end < start:
        raise ValueError("A usage log line ends before it starts.")
    return Entry(
        data["session_id"], data["account"], start, end, Unit(data["unit"]), _amount(data["amount"])
    )


def _amount(value: object) -> Decimal:
    """An amount from its decimal text; anything else refuses."""
    try:
        amount = Decimal(value) if isinstance(value, str) else Decimal("nan")
    except InvalidOperation:
        amount = Decimal("nan")
    if not amount.is_finite() or amount < 0:
        raise ValueError("A usage log amount must be nonnegative decimal text.")
    return amount
