"""The usage log keeps exact amounts and notices any edit, truncation or repeat."""

import hashlib
import json
from decimal import Decimal

import pytest

from forge.core.quota import Unit
from forge.ledger import append, read


def session(**changes):
    data = {
        "session_id": "forge-run-1",
        "account": "kaggle",
        "gpu": "T4",
        "gpu_count": 2,
        "start": "2026-10-05T08:00:00Z",
        "end": "2026-10-05T10:00:00Z",
        "unit": "gpu-hours",
        "amount": "4.000000000",
        "run": "run",
        "job": "project",
        "job_digest": "0123456789ab",
    }
    data.update(changes)
    return data


def test_each_line_hashes_the_one_before_and_amounts_stay_exact(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.touch()
    append(log, session())
    append(log, session(session_id="forge-run-2", unit="usd", amount="0.1"))
    first, second = log.read_bytes().splitlines(keepends=True)
    assert json.loads(first)["prev_sha256"] is None
    assert json.loads(second)["prev_sha256"] == hashlib.sha256(first).hexdigest()
    entries = read(log)
    assert [entry.amount for entry in entries] == [Decimal("4.000000000"), Decimal("0.1")]
    assert entries[1].unit is Unit.USD


def test_an_edited_or_half_written_log_refuses_instead_of_giving_free_time(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.touch()
    append(log, session())
    append(log, session(session_id="forge-run-2"))
    original = log.read_bytes()
    log.write_bytes(original.replace(b'"amount": "4.000000000"', b'"amount": "3.000000000"', 1))
    with pytest.raises(ValueError, match="hash chain"):
        read(log)
    log.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="half-written"):
        read(log)


def test_a_session_is_recorded_once(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.touch()
    append(log, session())
    with pytest.raises(ValueError, match="already in the usage log"):
        append(log, session())
    assert len(read(log)) == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"amount": 4}, "decimal text"),
        ({"amount": "-1"}, "decimal text"),
        ({"amount": "NaN"}, "decimal text"),
        ({"gpu_count": 0}, "GPU count"),
        ({"end": "2026-10-05T07:00:00Z"}, "ends before it starts"),
        ({"start": "2026-10-05T08:00:00"}, "no time zone"),
        ({"extra": True}, "unknown fields"),
        ({"run": ""}, "empty name"),
    ],
)
def test_invalid_lines_refuse_before_anything_is_written(tmp_path, changes, message):
    log = tmp_path / "usage.jsonl"
    log.touch()
    with pytest.raises(ValueError, match=message):
        append(log, session(**changes))
    assert log.read_bytes() == b""
