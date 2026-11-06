"""The run loop: place a session, start it, follow it, and when it ends, continue elsewhere.

Each turn of the loop chooses an account, holds its time, starts a session, streams its
log until it ends, fetches its results and records its usage. A paused session continues
on the next account from its snapshot; a finished one hands back its outputs.
"""

from __future__ import annotations

import os
import shutil
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from forge import accounting, accounts, files, jobs, runs
from forge.accounts import Account
from forge.core.placement import Placement, Request, choose, mismatch, next_reset
from forge.home import Home
from forge.providers import Provider, State, for_account
from forge.providers.bundle import SessionSpec

POLL_SECONDS = {"local": 0.5, "kaggle": 30.0, "modal": 10.0}
OUTCOMES = {State.FINISHED: runs.FINISHED, State.PAUSED: runs.PAUSED, State.FAILED: runs.FAILED}
LOG_TAIL_LINES = 20


class Detached(Exception):  # noqa: N818 - a signal for the caller, not an error
    """Raised when the user pressed Ctrl-C: the session keeps running without us."""


def start(home: Home, job: jobs.Job, request: Request, name: str) -> str:
    """Create a run for ``job`` and return its name. Nothing starts until ``follow``.

    Raises:
        ValueError: When no plugged-in account offers the GPU the request names.
    """
    candidates = accounting.candidates(home, accounts.load(home.accounts), accounting.now())
    if (reason := mismatch(candidates, request)) is not None:
        raise ValueError(reason)
    folder = runs.folder(home.runs, name)
    folder.mkdir(parents=True)
    digest = jobs.pack(job, folder / "job.zip")
    journal = {
        "version": runs.JOURNAL_VERSION,
        "run": name,
        "state": runs.RUNNING,
        "job": {
            "folder": str(job.folder),
            "script": job.script,
            "args": job.args,
            "requirements": job.requirements,
            "digest": digest,
            "declaration": asdict(job.declaration),
        },
        "request": {"seconds": request.seconds, "gpu": request.gpu, "memory_gb": request.memory_gb},
        "sessions": [],
    }
    runs.write(folder / "journal.json", journal)
    return name


def follow(home: Home, run: str, out: TextIO, on: str | None = None) -> int:
    """Drive the run until it finishes, fails, pauses or has nowhere to go; return a status.

    Several forge commands can follow one run at once. Starting a session and closing one
    each happen under the run's lock, on a freshly read journal, so each happens once. A
    background follower takes its own process ID out of the journal when it is done.
    """
    path = runs.journal_path(home.runs, run)
    try:
        return _follow(home, run, out, on)
    finally:
        with runs.locked(path):
            journal = runs.read(path)
            if journal.get("follower") == os.getpid():
                del journal["follower"]
                runs.write(path, journal)


def _follow(home: Home, run: str, out: TextIO, on: str | None) -> int:
    """The loop ``follow`` runs: close the session that ended, start the next, repeat."""
    path = runs.journal_path(home.runs, run)
    while True:
        with runs.locked(path):
            journal = runs.read(path)
            session = runs.active(journal)
            if session is None:
                if journal["state"] != runs.RUNNING:
                    return _report(home, journal, out)
                session = _begin(home, journal, out, on)
                on = None
                if session is None:
                    return 0
        state, shown = _watch(home, run, session, out)
        with runs.locked(path):
            _end(home, runs.read(path), session["number"], state)
        # Kaggle hands back a notebook's log only once it ends: show what was not streamed.
        out.write(_saved_log(runs.folder(home.runs, run), session["number"])[shown:])


def _begin(
    home: Home, journal: dict[str, Any], out: TextIO, on: str | None
) -> dict[str, Any] | None:
    """Choose an account and start the next session, or mark the run as waiting.

    Placement and the hold happen under one lock for all runs, so two runs that start at
    the same moment cannot both plan on the same free time. An account that cannot start
    the session cools down, and the next account is tried.
    """
    known = accounts.load(home.accounts)
    path = _journal_path(home, journal)
    failures: list[str] = []
    while True:
        with files.locked(home.placing):
            placement = _place(home, journal, known, on)
            if placement is None:
                _wait(home, journal, known, out, failures)
                return None
            account = accounts.find(known, placement.account)
            session = _new_session(journal, placement)
            session["hold"] = accounting.hold(account, session["gpu"], session["seconds"])
            runs.write(path, journal)
        out.write(f"Session {session['number']} on {account.name} ({_limit(placement)}).\n")
        reason = _start(home, journal, account, session)
        if reason is None:
            session["phase"] = runs.ACTIVE
            runs.write(path, journal)
            return session
        out.write(f"{account.name} could not start a session: {reason}. Trying the next account.\n")
        failures.append(f"{account.name} ({reason})")
        on = None


def _start(
    home: Home, journal: dict[str, Any], account: Account, session: dict[str, Any]
) -> str | None:
    """Start the session; if the provider refuses, close it and return why, else None.

    Whatever did start is stopped first. A session the provider still shows running keeps
    its hold, and the refusal is raised as before.
    """
    spec = _spec(home, journal, session)
    provider = _provider(home, account)
    try:
        provider.start(account, spec)
    except ValueError as error:
        if not _nothing_left(provider, account, session["handle"]):
            raise
        reason = str(error).splitlines()[0] if str(error) else type(error).__name__
        reason = reason.removeprefix(f"{account.name}: ").rstrip(".")
        session["end"] = runs.now_text()
        session["outcome"] = runs.FAILED
        session["reason"] = reason
        session["phase"] = runs.RECORDED
        runs.write(_journal_path(home, journal), journal)
        accounting.cool_down(home, account.name, accounting.now())
        return reason
    return None


def _nothing_left(provider: Provider, account: Account, handle: str) -> bool:
    """Stop what a failed start left behind; False only if the provider shows it running.

    A provider that cannot find the session at all has nothing running for it.
    """
    try:
        return provider.stop(account, handle)
    except ValueError:
        return True


def _place(
    home: Home, journal: dict[str, Any], known: list[Account], on: str | None
) -> Placement | None:
    """Ask placement for the next account, avoiding a tie with the one just used."""
    if on is not None:
        accounts.find(known, on)
    previous = journal["sessions"][-1]["account"] if journal["sessions"] else None
    request = Request(
        seconds=journal["request"]["seconds"],
        gpu=journal["request"].get("gpu"),
        memory_gb=journal["request"].get("memory_gb"),
        on=on,
        previous=previous,
    )
    return choose(accounting.candidates(home, known, accounting.now()), request)


def _new_session(journal: dict[str, Any], placement: Placement) -> dict[str, Any]:
    """Add the next session to the journal, named before anything is started."""
    number = len(journal["sessions"]) + 1
    session = {
        "number": number,
        "account": placement.account,
        "gpu": placement.gpu,
        "handle": f"forge-{journal['run']}-{number}",
        "phase": runs.STARTING,
        "start": runs.now_text(),
        "seconds": placement.seconds,
        "hold": None,
    }
    journal["sessions"].append(session)
    return session


def _spec(home: Home, journal: dict[str, Any], session: dict[str, Any]) -> SessionSpec:
    """What the provider needs to start the session, including the snapshot to resume."""
    folder = _run_folder(home, journal)
    job = journal["job"]
    snapshot = _latest_snapshot(folder, journal)
    described = jobs.Job(Path(job["folder"]), job["script"], job["args"], job["requirements"])
    return SessionSpec(
        run=journal["run"],
        number=session["number"],
        archive=folder / "job.zip",
        script=job["script"],
        args=job["args"],
        requirements=job["requirements"],
        resume=snapshot,
        seconds=session["seconds"],
        gpu=session["gpu"],
        torch=jobs.session_torch(described, snapshot, _installed_torch()),
    )


def _watch(home: Home, run: str, session: dict[str, Any], out: TextIO) -> tuple[State, int]:
    """Stream the session's log until it stops running. Ctrl-C detaches instead.

    Returns how the session ended and how many characters of its log were shown.
    """
    account = accounts.find(accounts.load(home.accounts), session["account"])
    provider = _provider(home, account)
    shown = 0
    try:
        while True:
            state = provider.poll(account, session["handle"])
            text = provider.logs(account, session["handle"])
            out.write(text[shown:])
            out.flush()
            shown = len(text)
            if state not in (State.RUNNING, State.UNKNOWN):
                return state, shown
            time.sleep(POLL_SECONDS[account.provider])
    except KeyboardInterrupt:
        raise Detached(run) from None


def _end(home: Home, journal: dict[str, Any], number: int, state: State) -> None:
    """Fetch the session's results, confirm it stopped, record its usage, update the run.

    Does nothing if another forge has already closed session ``number``.
    """
    session = runs.active(journal)
    if session is None or session["number"] != number:
        return
    account = accounts.find(accounts.load(home.accounts), session["account"])
    provider = _provider(home, account)
    destination = _run_folder(home, journal) / "sessions" / str(session["number"])
    if not destination.exists():
        provider.fetch(account, session["handle"], destination)
    if not provider.stop(account, session["handle"]):
        raise ValueError(f"{account.name} has not confirmed that {session['handle']} stopped.")
    session["end"] = session.get("end") or runs.now_text()
    session["outcome"] = OUTCOMES.get(state, runs.FAILED)
    runs.write(_journal_path(home, journal), journal)
    accounting.record_session(home, journal, account)
    session["phase"] = runs.RECORDED
    if session["outcome"] == runs.FINISHED:
        _unpack_outputs(destination, _run_folder(home, journal) / "outputs")
    # A paused session leaves the run as it was: running continues, paused by you waits.
    if session["outcome"] != runs.PAUSED:
        journal["state"] = session["outcome"]
    runs.write(_journal_path(home, journal), journal)


def _wait(
    home: Home, journal: dict[str, Any], known: list[Account], out: TextIO, failures: list[str]
) -> None:
    """Mark the run as waiting and say why: which accounts failed, when free time returns."""
    journal["state"] = runs.WAITING
    runs.write(_journal_path(home, journal), journal)
    if failures:
        minutes = accounting.COOLDOWN_SECONDS // 60
        out.write(
            f"No account could start this run: {'; '.join(failures)}. "
            f"Each is passed over for {minutes} minutes.\n"
        )
    reset = next_reset(accounting.candidates(home, known, accounting.now()))
    when = f" The next allowance resets at {reset.isoformat()}." if reset else ""
    out.write(f"No account has time for this run now.{when}\n")
    out.write(f"Continue it later with: forge resume {journal['run']}\n")


def _report(home: Home, journal: dict[str, Any], out: TextIO) -> int:
    """Say how the run ended and return its exit status."""
    state = journal["state"]
    folder = _run_folder(home, journal)
    count = len(journal["sessions"])
    sessions = "1 session" if count == 1 else f"{count} sessions"
    if state == runs.FINISHED:
        out.write(
            f"Run {journal['run']} finished after {sessions}. Outputs: {folder / 'outputs'}\n"
        )
        return 0
    if state == runs.PAUSED:
        out.write(f"Run {journal['run']} is paused. Continue with: forge resume {journal['run']}\n")
        return 0
    out.write(f"Run {journal['run']} {state}. Last lines of its log:\n{_tail(folder, journal)}")
    return 1


def _tail(folder: Path, journal: dict[str, Any]) -> str:
    """The end of the last session's log."""
    if not journal["sessions"]:
        return ""
    lines = _saved_log(folder, journal["sessions"][-1]["number"]).splitlines()
    return "".join(f"  {line}\n" for line in lines[-LOG_TAIL_LINES:])


def _saved_log(folder: Path, number: int) -> str:
    """The log a closed session left in the run's folder, or nothing if it left none."""
    log = folder / "sessions" / str(number) / "log.txt"
    return log.read_text(errors="replace") if log.exists() else ""


def _latest_snapshot(folder: Path, journal: dict[str, Any]) -> Path | None:
    """The snapshot of the most recent paused session, if the run has one."""
    for session in reversed(journal["sessions"]):
        snapshot = folder / "sessions" / str(session["number"]) / "snapshot"
        if session.get("outcome") == runs.PAUSED and snapshot.is_dir():
            return snapshot
    return None


def _unpack_outputs(session: Path, outputs: Path) -> None:
    """Extract the finished session's outputs into the run's ``outputs`` folder."""
    if outputs.exists():
        shutil.rmtree(outputs)
    outputs.mkdir()
    archive = session / "outputs.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(outputs)


def _limit(placement: Placement) -> str:
    """How long the session may run, in words."""
    return f"up to {placement.seconds} s" if placement.seconds is not None else "no time limit"


def _provider(home: Home, account: Account) -> Provider:
    """The provider for an account, with local sessions kept in forge's state folder."""
    return for_account(account, home.state / "sessions")


def _run_folder(home: Home, journal: dict[str, Any]) -> Path:
    """The run's folder in forge's state."""
    return runs.folder(home.runs, journal["run"])


def _journal_path(home: Home, journal: dict[str, Any]) -> Path:
    """Where the run's journal is written."""
    return _run_folder(home, journal) / "journal.json"


def _installed_torch() -> str:
    """The torch version installed with forge, read without importing torch."""
    from importlib.metadata import version  # noqa: PLC0415 - only needed here

    return version("torch")
