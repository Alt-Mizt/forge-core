"""A run hops between accounts at each session limit and ends with the same weights."""

import io
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from forge import accounts, control, jobs, loop, runs
from forge.core.placement import Request
from forge.home import Home

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/tiny/train.py"
SESSION_SECONDS = 10
STARTUP_LIMIT_SECONDS = 120
POLL_SECONDS = 0.1


@pytest.fixture(scope="module", autouse=True)
def warm_torch():
    """Import torch once, so a cold disk cache cannot eat a whole 10-second session."""
    subprocess.run([sys.executable, "-c", "import torch"], check=True)


@pytest.fixture
def home(tmp_path):
    place = Home(tmp_path / "config", tmp_path / "state")
    place.prepare()
    return place


def add_local(home, *names, session_limit: int | None = SESSION_SECONDS):
    added = [
        accounts.new_account("local", name=name, device="cpu", session_limit=session_limit)
        for name in names
    ]
    accounts.save(home.accounts, accounts.load(home.accounts) + added)


def uninterrupted_hash(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True, cwd=tmp_path
    )
    return result.stdout.strip()


def test_a_run_hops_three_times_and_matches_an_uninterrupted_run(home, tmp_path):
    add_local(home, "first", "second")
    job = jobs.describe(SCRIPT, ["--step-delay", "1"])
    loop.start(home, job, Request(), "hop")
    out = io.StringIO()
    assert loop.follow(home, "hop", out) == 0
    journal = runs.read(home.runs / "hop" / "journal.json")
    sessions = journal["sessions"]
    assert len(sessions) >= 4, out.getvalue()
    assert {session["account"] for session in sessions} == {"first", "second"}
    assert [session["outcome"] for session in sessions[:-1]] == ["paused"] * (len(sessions) - 1)
    final = (home.runs / "hop" / "outputs" / "weights-sha256.txt").read_text().strip()
    assert final == uninterrupted_hash(tmp_path)
    assert len(home.usage.read_text().splitlines()) == len(sessions)


def test_with_no_account_available_the_run_waits_and_says_how_to_resume(home):
    loop.start(home, jobs.describe(SCRIPT, []), Request(), "lonely")
    out = io.StringIO()
    assert loop.follow(home, "lonely", out) == 0
    assert "forge resume lonely" in out.getvalue()
    assert runs.read(home.runs / "lonely" / "journal.json")["state"] == runs.WAITING


def test_pause_keeps_a_snapshot_and_resume_finishes_the_run(home, tmp_path):
    add_local(home, "desk", session_limit=None)
    progress = tmp_path / "progress"
    arguments = ["--step-delay", "0.5", "--progress", str(progress)]
    loop.start(home, jobs.describe(SCRIPT, arguments), Request(), "paused")
    journal_path = home.runs / "paused" / "journal.json"
    started = loop._begin(home, runs.read(journal_path), io.StringIO(), None)
    assert started is not None
    wait_for_first_step(progress)
    assert control.pause(home, "paused", io.StringIO()) == 0
    journal = runs.read(journal_path)
    assert journal["state"] == runs.PAUSED and journal["sessions"][-1]["outcome"] == "paused"
    assert control.resume(home, "paused", io.StringIO(), None) == 0
    final = (home.runs / "paused" / "outputs" / "weights-sha256.txt").read_text().strip()
    assert final == uninterrupted_hash(tmp_path)


def test_a_pause_from_elsewhere_closes_a_followed_session_exactly_once(home, tmp_path):
    add_local(home, "desk", session_limit=None)
    progress = tmp_path / "progress"
    arguments = ["--step-delay", "0.5", "--progress", str(progress)]
    loop.start(home, jobs.describe(SCRIPT, arguments), Request(), "watched")
    follower = threading.Thread(target=loop.follow, args=(home, "watched", io.StringIO()))
    follower.start()
    wait_for_first_step(progress)
    assert control.pause(home, "watched", io.StringIO()) == 0
    follower.join(timeout=STARTUP_LIMIT_SECONDS)
    assert not follower.is_alive()
    journal = runs.read(home.runs / "watched" / "journal.json")
    assert journal["state"] == runs.PAUSED
    assert [session["outcome"] for session in journal["sessions"]] == ["paused"]
    assert len(home.usage.read_text().splitlines()) == 1


def test_down_stops_a_live_session_and_records_it(home):
    add_local(home, "desk", session_limit=None)
    loop.start(home, jobs.describe(SCRIPT, ["--step-delay", "5"]), Request(), "stopped")
    loop._begin(home, runs.read(home.runs / "stopped" / "journal.json"), io.StringIO(), None)
    out = io.StringIO()
    assert control.down(home, None, out) == 0
    assert "confirms session 1 stopped" in out.getvalue()
    journal = runs.read(home.runs / "stopped" / "journal.json")
    assert journal["state"] == runs.STOPPED
    assert journal["sessions"][-1]["phase"] == runs.RECORDED
    assert len(home.usage.read_text().splitlines()) == 1


def wait_for_first_step(progress):
    """Wait until the job has trained a step, which it records in its progress file."""
    deadline = time.monotonic() + STARTUP_LIMIT_SECONDS
    while not progress.exists():
        assert time.monotonic() < deadline, "the job never trained a step"
        time.sleep(POLL_SECONDS)
