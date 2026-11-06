"""Run sessions on this machine, each in its own folder, as a detached process.

A session keeps running if forge itself exits; ``forge resume`` picks it up again. The
session runner enforces the hard time limit, so it holds even when forge is not watching.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from forge.accounts import Account
from forge.jobs import plain_version
from forge.processes import alive, end_groups
from forge.providers import State, state_from_exit
from forge.providers.bundle import SessionSpec, build, unpack


class Local:
    """Sessions on this machine. Each one lives in ``sessions/<handle>``."""

    def __init__(self, sessions: Path) -> None:
        """Keep session folders under ``sessions``."""
        self.sessions = sessions

    def check(self, account: Account) -> str:
        """Report the device and torch version this machine offers."""
        import torch  # noqa: PLC0415 - torch is slow to import, so only when needed

        return f"{account.device or 'cpu'}, torch {plain_version(torch.__version__)}"

    def start(self, account: Account, spec: SessionSpec) -> None:
        """Unpack the session folder and start the session runner in the background."""
        _require_torch(spec.torch)
        folder = self.sessions / spec.handle
        unpack(build(spec), folder)
        command = [sys.executable, "-m", "forge.session", str(folder)]
        if spec.seconds is not None:
            command.append(str(spec.seconds))
        environment = dict(os.environ, PYTHONPATH=str(folder), FORGE_DEVICE=account.device or "cpu")
        with (folder / "session.log").open("wb") as log:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=log, start_new_session=True
            )
        pending = folder / "session.pid.pending"
        pending.write_text(f"{process.pid}\n")
        pending.replace(folder / "session.pid")

    def poll(self, account: Account, handle: str) -> State:
        """Finished, paused or failed once the runner wrote its exit code; else running."""
        del account
        out = self.sessions / handle / "out"
        if (out / "exit_code").exists():
            return state_from_exit(int((out / "exit_code").read_text()))
        if alive(_pid(self.sessions / handle / "session.pid")):
            return State.RUNNING
        return State.FAILED

    def fetch(self, account: Account, handle: str, destination: Path) -> None:
        """Copy the session's ``out`` folder; a session that never started leaves it empty."""
        del account
        out = self.sessions / handle / "out"
        if out.is_dir():
            shutil.copytree(out, destination)
        else:
            destination.mkdir(parents=True)

    def logs(self, account: Account, handle: str) -> str:
        """The job's output so far."""
        del account
        log = self.sessions / handle / "out" / "log.txt"
        return log.read_text(errors="replace") if log.exists() else ""

    def pause(self, account: Account, handle: str) -> None:
        """Send SIGTERM to the wrapped job, which saves after its current batch."""
        del account
        pid = _pid(self.sessions / handle / "out" / "runner.pid")
        if pid is None:
            raise ValueError("The session has not started its job yet; try again in a moment.")
        os.kill(pid, signal.SIGTERM)

    def stop(self, account: Account, handle: str) -> bool:
        """Kill the job and its session runner, then wait until both are gone."""
        del account
        folder = self.sessions / handle
        stored = (_pid(folder / "out" / "runner.pid"), _pid(folder / "session.pid"))
        # Each process leads its own group, so this also ends any data workers.
        return end_groups([pid for pid in stored if pid is not None], signal.SIGKILL)


def _require_torch(version: str) -> None:
    """Refuse a session that needs a different torch than the one installed here."""
    import torch  # noqa: PLC0415 - torch is slow to import, so only when needed

    installed = plain_version(torch.__version__)
    if installed != version:
        raise ValueError(
            f"This run needs torch {version}, but this machine has torch {installed}. "
            f"Install torch=={version} here, or continue the run on a cloud account."
        )


def _pid(path: Path) -> int | None:
    """The process ID stored in ``path``, if the file exists."""
    return int(path.read_text()) if path.exists() else None
