"""Where forge keeps its configuration and state, outside any project folder.

``FORGE_HOME`` puts everything in one folder, which tests and demos use. Without it, the
account list follows the XDG config location and all state the XDG data location.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Home:
    """The two folders forge uses: one for account configuration, one for state."""

    config: Path
    state: Path

    @property
    def accounts(self) -> Path:
        """The account list, ``accounts.toml``."""
        return self.config / "accounts.toml"

    @property
    def usage(self) -> Path:
        """The usage log: one line per finished session, each hashing the line before."""
        return self.state / "usage.jsonl"

    @property
    def checks(self) -> Path:
        """The result of the last ``forge check`` for each account."""
        return self.state / "checks.json"

    @property
    def cooldowns(self) -> Path:
        """Accounts that could not start a session lately, and when each may be tried again."""
        return self.state / "cooldowns.json"

    @property
    def runs(self) -> Path:
        """One folder per run: its journal, job archive, snapshots, logs and outputs."""
        return self.state / "runs"

    @property
    def placing(self) -> Path:
        """The lock held while a session is placed, so two runs never hold the same time."""
        return self.state / "placing.lock"

    def prepare(self) -> None:
        """Create the folders, and an empty usage log, on first use."""
        for folder in (self.config, self.state, self.runs):
            folder.mkdir(parents=True, exist_ok=True)
        self.usage.touch(exist_ok=True)


def locate() -> Home:
    """Find forge's home from ``FORGE_HOME``, else from the XDG locations."""
    if root := os.environ.get("FORGE_HOME"):
        return Home(Path(root), Path(root) / "state")
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Home(Path(config) / "forge", Path(data) / "forge")
