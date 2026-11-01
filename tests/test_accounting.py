"""Holds come from the run journals, recorded sessions from the usage log, and they add up."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from forge import accounting, accounts, jobs, loop, runs
from forge.core.placement import Request
from forge.home import Home

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/tiny/train.py"
NOW = datetime(2026, 10, 7, 12, tzinfo=UTC)
HOUR = 3600


def home_with(tmp_path, *plugged):
    home = Home(tmp_path / "config", tmp_path / "state")
    home.prepare()
    accounts.save(home.accounts, list(plugged))
    return home


def start_session(home, account, seconds, hold=None):
    """A run whose first session has started on ``account`` and is not recorded yet."""
    name = f"run-{len(runs.all_journals(home.runs)) + 1}"
    loop.start(home, jobs.describe(SCRIPT, []), Request(), name)
    path = home.runs / name / "journal.json"
    journal = runs.read(path)
    journal["sessions"].append(
        {
            "number": 1,
            "account": account.name,
            "handle": f"forge-{name}-1",
            "phase": runs.ACTIVE,
            "start": "2026-10-07T11:00:00Z",
            "seconds": seconds,
            "hold": hold if hold is not None else accounting.hold(account, seconds),
        }
    )
    runs.write(path, journal)
    return journal


def test_a_running_sessions_hold_comes_off_the_time_left_until_it_is_recorded(tmp_path):
    kaggle = accounts.new_account("kaggle")
    home = home_with(tmp_path, kaggle)
    journal = start_session(home, kaggle, 2 * HOUR)
    assert journal["sessions"][0]["hold"] == "2.000000000"
    assert accounting.remaining(home, [kaggle], NOW)["kaggle"].amount == Decimal(28)
    assert accounting.candidates(home, [kaggle], NOW)[0].busy

    journal["sessions"][0]["end"] = "2026-10-07T11:30:00Z"
    accounting.record_session(home, journal, kaggle)
    journal["sessions"][0]["phase"] = runs.RECORDED
    runs.write(home.runs / journal["run"] / "journal.json", journal)
    # Half an hour was used, so the other hour and a half comes back.
    assert accounting.remaining(home, [kaggle], NOW)["kaggle"].amount == Decimal("29.5")
    assert not accounting.candidates(home, [kaggle], NOW)[0].busy


def test_recorded_use_never_exceeds_what_was_held(tmp_path):
    kaggle = accounts.new_account("kaggle")
    home = home_with(tmp_path, kaggle)
    journal = start_session(home, kaggle, HOUR)
    journal["sessions"][0]["end"] = "2026-10-07T14:00:00Z"  # the provider ran long
    accounting.record_session(home, journal, kaggle)
    accounting.record_session(home, journal, kaggle)  # a second call changes nothing
    assert len(home.usage.read_text().splitlines()) == 1
    assert accounting.remaining(home, [kaggle], NOW)["kaggle"].amount == Decimal(28)


def test_modal_holds_the_collection_time_too_and_counts_dollars(tmp_path):
    modal = accounts.new_account("modal")
    assert accounting.hold(modal, 150) == "0.270"  # (150 + 120) seconds at $0.001
    home = home_with(tmp_path, modal)
    start_session(home, modal, 150)
    assert accounting.remaining(home, [modal], NOW)["modal"].amount == Decimal("29.730")


def test_a_local_account_has_no_allowance_and_holds_nothing(tmp_path):
    local = accounts.new_account("local", device="cpu")
    home = home_with(tmp_path, local)
    assert accounting.hold(local, 10) is None
    assert accounting.remaining(home, [local], NOW) == {}
    [candidate] = accounting.candidates(home, [local], NOW)
    assert candidate.local and candidate.free_seconds is None and not candidate.has_gpu
