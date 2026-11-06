"""The commands that act on existing runs: detach, pause, resume, down, status and logs."""

from __future__ import annotations

import signal
import subprocess
import sys
from typing import Any, TextIO

from forge import accounting, accounts, loop, runs
from forge.home import Home
from forge.ledger import timestamp
from forge.processes import alive, end_groups
from forge.providers import for_account


def detach(home: Home, run: str, out: TextIO, on: str | None) -> int:
    """Start a background follower that carries the run from session to session.

    The follower is ``forge resume RUN`` in its own process group, writing to the run's
    ``follow.log``. Its process ID goes into the journal before it can read the journal.
    """
    path = runs.journal_path(home.runs, run)
    command = [sys.executable, "-m", "forge.cli", "resume", run]
    if on is not None:
        command += ["--on", on]
    with runs.locked(path), (path.parent / "follow.log").open("ab") as log:
        follower = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True
        )
        journal = runs.read(path)
        journal["follower"] = follower.pid
        runs.write(path, journal)
    out.write(
        f"Run {run} started in the background. "
        f"forge status, forge logs {run}, forge resume {run} to attach.\n"
    )
    return 0


def pause(home: Home, run: str, out: TextIO) -> int:
    """Ask the running session to save and stop, then fetch its snapshot."""
    path = runs.journal_path(home.runs, run)
    with runs.locked(path):
        journal = runs.read(path)
        if journal["state"] in runs.ENDED_STATES:
            out.write(f"Run {run} has already {journal['state']}.\n")
            return 1
        journal["state"] = runs.PAUSED
        runs.write(path, journal)
        session = runs.active(journal)
    if session is not None:
        account = accounts.find(accounts.load(home.accounts), session["account"])
        for_account(account, home.state / "sessions").pause(account, session["handle"])
        out.write(f"Asked session {session['number']} on {account.name} to save and stop.\n")
    return loop.follow(home, run, out)


def resume(home: Home, run: str, out: TextIO, on: str | None) -> int:
    """Continue a paused or waiting run, or follow one that is still running."""
    path = runs.journal_path(home.runs, run)
    with runs.locked(path):
        journal = runs.read(path)
        if journal["state"] in runs.ENDED_STATES:
            out.write(f"Run {run} has already {journal['state']}.\n")
            return 1
        journal["state"] = runs.RUNNING
        runs.write(path, journal)
    return loop.follow(home, run, out, on)


def down(home: Home, run: str | None, out: TextIO) -> int:
    """Stop the named run, or every run with a live session, and confirm each shutdown."""
    names = [run] if run else [journal["run"] for journal in runs.all_journals(home.runs)]
    for name in names:
        path = runs.journal_path(home.runs, name)
        _stop_follower(runs.read(path), out)
        with runs.locked(path):
            journal = runs.read(path)
            journal.pop("follower", None)
            session = runs.active(journal)
            if session is not None:
                _stop(home, journal, session, out)
            elif run is None:
                continue
            if journal["state"] not in runs.ENDED_STATES:
                journal["state"] = runs.STOPPED
            runs.write(path, journal)
    out.write("Nothing forge started is still running.\n")
    return 0


def status(home: Home) -> list[dict[str, str | int]]:
    """One row per run: its name, state, last account and the time its sessions took."""
    rows = []
    for journal in runs.all_journals(home.runs):
        sessions = journal["sessions"]
        rows.append(
            {
                "run": journal["run"],
                "state": journal["state"],
                "account": sessions[-1]["account"] if sessions else "-",
                "sessions": len(sessions),
                "seconds": sum(_seconds(session) for session in sessions),
            }
        )
    return rows


def logs(home: Home, run: str) -> str:
    """The log of every session of a run, the live one included."""
    folder = runs.folder(home.runs, run)
    journal = runs.read(runs.journal_path(home.runs, run))
    parts = []
    for session in journal["sessions"]:
        parts.append(f"--- session {session['number']} on {session['account']}\n")
        if "reason" in session:
            parts.append(f"Could not start: {session['reason']}\n")
            continue
        saved = folder / "sessions" / str(session["number"]) / "log.txt"
        if saved.exists():
            parts.append(saved.read_text(errors="replace"))
        elif session["phase"] != runs.RECORDED:
            account = accounts.find(accounts.load(home.accounts), session["account"])
            provider = for_account(account, home.state / "sessions")
            parts.append(provider.logs(account, session["handle"]))
    return "".join(parts)


def _stop_follower(journal: dict[str, Any], out: TextIO) -> None:
    """End the run's background follower, if one still runs, and confirm it is gone.

    A process ID can be reused once its process has ended, so only a process that is
    still this run's ``forge resume`` is ended.
    """
    pid = journal.get("follower")
    if pid is None or not alive(pid) or not _follows(pid, journal["run"]):
        return
    if not end_groups([pid], signal.SIGTERM):
        raise ValueError(f"The follower of {journal['run']} (process {pid}) has not stopped.")
    out.write(f"{journal['run']}: its background follower stopped.\n")


def _follows(pid: int, run: str) -> bool:
    """Whether process ``pid`` is ``forge resume`` for this run, by its command line."""
    shown = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return f"forge.cli resume {run}" in shown.stdout


def _stop(home: Home, journal: dict[str, Any], session: dict[str, Any], out: TextIO) -> None:
    """Stop one live session, confirm it, and record its usage."""
    account = accounts.find(accounts.load(home.accounts), session["account"])
    if not for_account(account, home.state / "sessions").stop(account, session["handle"]):
        raise ValueError(
            f"{account.name} has not confirmed that {session['handle']} stopped; its hold is kept."
        )
    session["end"] = runs.now_text()
    session["outcome"] = runs.STOPPED
    runs.write(runs.folder(home.runs, journal["run"]) / "journal.json", journal)
    accounting.record_session(home, journal, account)
    session["phase"] = runs.RECORDED
    out.write(f"{journal['run']}: {account.name} confirms session {session['number']} stopped.\n")


def _seconds(session: dict[str, Any]) -> int:
    """How long a session ran, or has run so far."""
    end = session.get("end") or runs.now_text()
    return int((timestamp(end) - timestamp(session["start"])).total_seconds())
