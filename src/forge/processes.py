"""Processes forge starts on this machine: whether one still runs, and ending one for certain.

Every such process leads its own process group, so ending the group also ends anything it
started, such as a job's data workers or a follower's provider calls.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time

STOP_WAIT_SECONDS = 10
POLL_SECONDS = 0.1


def alive(pid: int | None) -> bool:
    """Whether a process still runs, reaping it first if it is our own finished child."""
    if pid is None:
        return False
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
        if finished:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def end_groups(pids: list[int], sent: signal.Signals) -> bool:
    """Send ``sent`` to each process's group, then wait; True once every process is gone."""
    for pid in pids:
        if alive(pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, sent)
    deadline = time.monotonic() + STOP_WAIT_SECONDS
    while any(alive(pid) for pid in pids):
        if time.monotonic() > deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True
