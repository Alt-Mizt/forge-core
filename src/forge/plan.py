"""``forge plan``: the route a job would take across the accounts, before anything runs.

It reads the accounts as they stand once, runs placement forward (see ``forge.core.route``)
and prints the sessions in order and what each account would have left. Nothing is
started, held or written.
"""

from __future__ import annotations

from typing import TextIO

from forge import accounting, accounts
from forge.core.placement import Candidate, Placement, Request, mismatch
from forge.core.route import route
from forge.home import Home

SECONDS_PER_HOUR = 3600


def show(home: Home, request: Request, hours: float | None, out: TextIO) -> int:
    """Print the route for a job of ``hours`` (one session when unknown); return a status.

    Raises:
        ValueError: When no plugged-in account offers the GPU the request names.
    """
    now = accounting.now()
    candidates = accounting.candidates(home, accounts.load(home.accounts), now)
    if (reason := mismatch(candidates, request)) is not None:
        raise ValueError(reason)
    seconds = round(hours * SECONDS_PER_HOUR) if hours is not None else None
    planned = route(candidates, request, seconds, now)
    known = {candidate.name: candidate for candidate in candidates}
    out.write(f"{'Session':<9}{'Account':<12}{'GPU':<11}{'Up to':>10}  {'Time':<6}Why\n")
    for number, placement in enumerate(planned.sessions, start=1):
        limit = _length(placement.seconds)
        kind = "paid" if placement.paid else "free"
        why = _why(known[placement.account], placement)
        out.write(
            f"{number:<9}{placement.account:<12}{placement.gpu:<11}{limit:>10}  {kind:<6}{why}\n"
        )
    out.write(f"\n{'Account':<12}Left after these sessions\n")
    for candidate in planned.left:
        out.write(f"{candidate.name:<12}{_left(candidate)}\n")
    out.write(_ending(planned.sessions, seconds, hours, covered=planned.covered))
    return 0


def _length(seconds: int | None) -> str:
    """A session's length in words: seconds under an hour, else hours to a tenth."""
    if seconds is None:
        return "no limit"
    return f"{seconds} s" if seconds < SECONDS_PER_HOUR else f"{seconds / SECONDS_PER_HOUR:.1f} h"


def _why(candidate: Candidate, placement: Placement) -> str:
    """Why placement chose this account, in a few words."""
    if candidate.local:
        return "local"
    if placement.paid:
        return "paid time within the paid cap"
    if candidate.resets_at is not None:
        return f"resets {candidate.resets_at:%a %d %b %H:%M} UTC"
    if candidate.expires is not None:
        return f"expires {candidate.expires:%Y-%m-%d}"
    return "free time"


def _left(candidate: Candidate) -> str:
    """What an account would have left, free and paid, in its own unit."""
    if candidate.unit is None:
        return "unlimited"
    unit = accounts.UNIT_LABELS[candidate.unit]
    paid = f" and {candidate.paid:.2f} {unit} paid" if candidate.paid else ""
    return f"{candidate.free:.2f} {unit} free{paid}"


def _ending(
    sessions: tuple[Placement, ...], seconds: int | None, hours: float | None, *, covered: bool
) -> str:
    """The last line: whether the route covers the job, and what to do if it does not."""
    if seconds is None:
        return "\nThis is the next session only. Add --hours to see the whole route.\n"
    if covered:
        return f"\nThese {len(sessions)} sessions cover the job's {hours:g} hours.\n"
    planned = sum(placement.seconds or 0 for placement in sessions)
    return (
        f"\nNo account can take the last {_length(seconds - planned)} of the job. "
        "forge run would wait there until an allowance resets.\n"
    )
