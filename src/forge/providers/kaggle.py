"""Run sessions as private Kaggle notebooks on Kaggle's free T4 machine.

Each call goes through ``bridge``, so the Kaggle SDK and the account's token live only in a
short-lived worker process. Kaggle cannot signal or cancel a running notebook from here: a
session saves and ends at its own time limit.
"""

from __future__ import annotations

from pathlib import Path

from forge.accounts import Account
from forge.providers import State, state_from_exit
from forge.providers.bridge import call
from forge.providers.bundle import SessionSpec, build
from forge.providers.notebook import source

# Kaggle can take minutes to accept a notebook or hand back its files.
SLOW_CALL_SECONDS = 300


class Client:
    """Kaggle sessions for one account at a time."""

    def check(self, account: Account) -> str:
        """Read the account's free quota, which proves its token works."""
        return call(account, "check", {})["note"]

    def start(self, account: Account, spec: SessionSpec) -> None:
        """Submit the private notebook that runs the session."""
        if spec.seconds is None:
            raise ValueError("A Kaggle session needs a time limit.")
        code = source(build(spec), spec.seconds, spec.torch, spec.requirements)
        request = {"handle": spec.handle, "code": code, "seconds": spec.seconds}
        call(account, "start", request, timeout=SLOW_CALL_SECONDS)

    def poll(self, account: Account, handle: str) -> State:
        """Running until the notebook ends; then the job's own exit status decides."""
        result = call(account, "poll", {"handle": handle}, timeout=SLOW_CALL_SECONDS)
        if "exit_code" in result:
            return state_from_exit(result["exit_code"])
        return State.RUNNING if result["state"] == "running" else State.FAILED

    def fetch(self, account: Account, handle: str, destination: Path) -> None:
        """Download the session's results."""
        request = {"handle": handle, "directory": str(destination.resolve())}
        call(account, "fetch", request, timeout=SLOW_CALL_SECONDS)

    def logs(self, account: Account, handle: str) -> str:
        """Kaggle hands back a notebook's output only once it ends; ``fetch`` includes it."""
        del account, handle
        return ""

    def pause(self, account: Account, handle: str) -> None:
        """Refuse: a running Kaggle notebook cannot be signalled."""
        del account, handle
        raise ValueError("Kaggle cannot pause a running notebook; it saves at its time limit.")

    def stop(self, account: Account, handle: str) -> bool:
        """True once Kaggle shows the notebook ended and holding no GPU time."""
        return bool(call(account, "stop", {"handle": handle})["stopped"])
