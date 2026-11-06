"""A detached run is carried by a background follower, which forge down also stops."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from forge import runs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/tiny/train.py"
RUN_LIMIT_SECONDS = 300
POLL_SECONDS = 0.2


@pytest.fixture(scope="module", autouse=True)
def warm_torch():
    """Import torch once, so a cold disk cache cannot eat a whole 10-second session."""
    subprocess.run([sys.executable, "-c", "import torch"], check=True)


def forge(home, *arguments):
    environment = dict(os.environ, FORGE_HOME=str(home))
    command = [sys.executable, "-m", "forge.cli", *arguments]
    return subprocess.run(command, env=environment, capture_output=True, text=True, check=False)


def journal(home, run):
    return runs.read(home / "state" / "runs" / run / "journal.json")


def wait_until(condition, message):
    deadline = time.monotonic() + RUN_LIMIT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(POLL_SECONDS)


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def add_local(home, name, *limit):
    forge(home, "add", "local", "--name", name, "--device", "cpu", *limit)


def test_a_detached_run_hops_and_finishes_on_its_own(tmp_path):
    for name in ("first", "second"):
        add_local(tmp_path, name, "--session-limit", "10")
    started = forge(
        tmp_path, "run", str(SCRIPT), "--name", "away", "--detach", "--", "--step-delay", "1"
    )
    assert started.returncode == 0, started.stderr
    assert started.stdout == (
        "Run away started in the background. forge status, forge logs away, "
        "forge resume away to attach.\n"
    )
    follower = journal(tmp_path, "away")["follower"]
    wait_until(lambda: journal(tmp_path, "away")["state"] == runs.FINISHED, "never finished")
    wait_until(lambda: gone(follower), "the follower outlived its run")
    finished = journal(tmp_path, "away")
    assert "follower" not in finished
    assert {session["account"] for session in finished["sessions"]} == {"first", "second"}
    log = (tmp_path / "state" / "runs" / "away" / "follow.log").read_text()
    assert "Run away finished after" in log


def test_down_stops_the_follower_before_the_session(tmp_path):
    add_local(tmp_path, "desk")
    started = forge(
        tmp_path, "run", str(SCRIPT), "--name", "held", "--detach", "--", "--step-delay", "5"
    )
    assert started.returncode == 0, started.stderr
    follower = journal(tmp_path, "held")["follower"]
    wait_until(
        lambda: (
            [session["phase"] for session in journal(tmp_path, "held")["sessions"]] == [runs.ACTIVE]
        ),
        "the session never started",
    )
    stopped = forge(tmp_path, "down", "held")
    assert stopped.returncode == 0, stopped.stderr
    assert "held: its background follower stopped." in stopped.stdout
    assert "confirms session 1 stopped" in stopped.stdout
    assert gone(follower)
    ended = journal(tmp_path, "held")
    assert ended["state"] == runs.STOPPED and "follower" not in ended
    assert ended["sessions"][-1]["phase"] == runs.RECORDED
