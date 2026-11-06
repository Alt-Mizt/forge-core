"""Run sessions in Modal sandboxes: one GPU, no network, gVisor, and a Modal-enforced timeout.

Each call goes through ``bridge``, so the Modal SDK and the account's token live only in a
short-lived worker process. By default a Modal account is used only while it has free
credit; set your workspace's spend limit in Modal too.
"""

from __future__ import annotations

import base64
from pathlib import Path

from forge.accounts import Account
from forge.providers import State, state_from_exit
from forge.providers.bridge import call
from forge.providers.bundle import SessionSpec, build

# Building the sandbox image the first time includes installing torch.
START_CALL_SECONDS = 900


class Client:
    """Modal sessions for one account at a time."""

    def check(self, account: Account) -> str:
        """Make an authenticated lookup, which proves the token works."""
        return call(account, "check", {})["note"]

    def start(self, account: Account, spec: SessionSpec) -> None:
        """Create the session's sandbox and give it the session folder."""
        if spec.seconds is None:
            raise ValueError("A Modal session needs a time limit.")
        request = {
            "handle": spec.handle,
            "bundle": base64.b64encode(build(spec)).decode(),
            "seconds": spec.seconds,
            "gpu": spec.gpu,
            "torch": spec.torch,
            "requirements": spec.requirements,
        }
        call(account, "start", request, timeout=START_CALL_SECONDS)

    def poll(self, account: Account, handle: str) -> State:
        """Running until the session wrote its exit status, which then decides."""
        result = call(account, "poll", {"handle": handle})
        if "exit_code" in result:
            return state_from_exit(result["exit_code"])
        return State.RUNNING if result["state"] == "running" else State.FAILED

    def fetch(self, account: Account, handle: str, destination: Path) -> None:
        """Write the session's result files into ``destination``."""
        files = call(account, "fetch", {"handle": handle})["files"]
        for name, content in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(content))
        destination.mkdir(parents=True, exist_ok=True)

    def logs(self, account: Account, handle: str) -> str:
        """The job's log so far."""
        return call(account, "logs", {"handle": handle})["text"]

    def pause(self, account: Account, handle: str) -> None:
        """Ask the job to save after its current batch."""
        call(account, "pause", {"handle": handle})

    def stop(self, account: Account, handle: str) -> bool:
        """Terminate the sandbox; True once Modal confirms it stopped."""
        return bool(call(account, "stop", {"handle": handle})["stopped"])
