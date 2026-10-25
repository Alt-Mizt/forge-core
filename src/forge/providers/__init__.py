"""One interface for every place a session can run: this machine, Kaggle and Modal.

Each provider can check an account, start a session with a hard time limit, poll it,
fetch its results, show its log, ask it to pause, and stop it with confirmed shutdown.
Cloud providers make every SDK call in a short-lived worker process (see ``bridge``).
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from forge.accounts import Account
    from forge.providers.bundle import SessionSpec

FINISHED_CODE, PAUSED_CODE = 0, 143


class State(Enum):
    """How a session stands, as far as its provider can tell."""

    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"
    UNKNOWN = "unknown"


class Provider(Protocol):
    """The operations forge needs from a provider. ``handle`` names one session."""

    def check(self, account: Account) -> str:
        """Confirm the credentials work with a cheap read-only call; return a short note."""
        ...

    def start(self, account: Account, spec: SessionSpec) -> None:
        """Start a session whose hard limit is ``spec.seconds``."""
        ...

    def poll(self, account: Account, handle: str) -> State:
        """Report the session's state."""
        ...

    def fetch(self, account: Account, handle: str, destination: Path) -> None:
        """Copy the ended session's results into ``destination`` (see forge.session)."""
        ...

    def logs(self, account: Account, handle: str) -> str:
        """Return the session's log so far."""
        ...

    def pause(self, account: Account, handle: str) -> None:
        """Ask the job to save a snapshot and stop at its next batch."""
        ...

    def stop(self, account: Account, handle: str) -> bool:
        """Stop the session now; return True only once the provider confirms it stopped."""
        ...


def for_account(account: Account, sessions: Path) -> Provider:
    """The provider for an account. Local sessions run in folders under ``sessions``."""
    if account.provider == "local":
        return import_module("forge.providers.local").Local(sessions)
    return import_module(f"forge.providers.{account.provider}").Client()


def state_from_exit(code: int) -> State:
    """Read a session's exit status: 0 finished, 143 paused, anything else failed."""
    if code == FINISHED_CODE:
        return State.FINISHED
    if code == PAUSED_CODE:
        return State.PAUSED
    return State.FAILED
