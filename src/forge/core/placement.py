"""Choose which account a job's next session runs on, and for how long.

The caller describes each account as a ``Candidate``; nothing here reads a file or a clock.
The order of preference is: a local machine that fits the job, then free allowances with
the soonest reset first, then paid time on accounts that allow it, cheapest first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

LOCAL, FREE, PAID = 0, 1, 2
# A session shorter than this cannot start a job, load a snapshot and save again.
MINIMUM_SECONDS = 1


@dataclass(frozen=True)
class Candidate:
    """One account as placement sees it.

    Attributes:
        name: The account name.
        local: Whether it is this machine.
        has_gpu: Whether it offers a GPU (a local account on CPU does not).
        usable: False when its credentials failed their last check.
        busy: True when it already runs a session and allows only one at a time.
        session_limit: Longest session in seconds, or None for no limit.
        free_seconds: Free time left, or None when unlimited.
        paid_seconds: Paid time left under the account's paid cap.
        price_per_second: What paid time costs, for ordering paid accounts.
        resets_at: When the free allowance starts again, if it has one.
    """

    name: str
    local: bool
    has_gpu: bool
    usable: bool
    busy: bool
    session_limit: int | None
    free_seconds: int | None
    paid_seconds: int
    price_per_second: Decimal
    resets_at: datetime | None


@dataclass(frozen=True)
class Request:
    """What the job asks for: a session length, a GPU, an account, and where it last ran."""

    seconds: int | None = None
    needs_gpu: bool = False
    on: str | None = None
    previous: str | None = None


@dataclass(frozen=True)
class Placement:
    """Where the next session runs, for how many seconds, and whether it uses paid time.

    ``seconds`` is None only for a local account with no limit: the job runs to the end.
    """

    account: str
    seconds: int | None
    paid: bool


def choose(candidates: list[Candidate], request: Request) -> Placement | None:
    """Return the best placement for the next session, or None when no account can take it."""
    options: list[tuple[tuple[int, float, bool, str], Candidate, int, int | None]] = []
    for candidate in candidates:
        if not _eligible(candidate, request):
            continue
        tier, seconds = _tier(candidate)
        seconds = _shortest(seconds, candidate.session_limit, request.seconds)
        if seconds is not None and seconds < MINIMUM_SECONDS:
            continue
        options.append((_rank(candidate, tier, request), candidate, tier, seconds))
    if not options:
        return None
    _, best, tier, seconds = min(options, key=lambda option: option[0])
    return Placement(best.name, seconds, paid=tier == PAID)


def next_reset(candidates: list[Candidate]) -> datetime | None:
    """The soonest time any account's free allowance starts again."""
    resets = [candidate.resets_at for candidate in candidates if candidate.resets_at]
    return min(resets) if resets else None


def _eligible(candidate: Candidate, request: Request) -> bool:
    """Whether the account may take this session at all, before comparing time left."""
    if request.on is not None and candidate.name != request.on:
        return False
    if not candidate.usable or candidate.busy:
        return False
    return candidate.has_gpu or not request.needs_gpu


def _tier(candidate: Candidate) -> tuple[int, int | None]:
    """Which kind of time the account would use next, and how many seconds of it are left."""
    if candidate.local:
        return LOCAL, candidate.free_seconds
    if candidate.free_seconds is None or candidate.free_seconds >= MINIMUM_SECONDS:
        return FREE, candidate.free_seconds
    return PAID, candidate.paid_seconds


def _shortest(*limits: int | None) -> int | None:
    """The smallest of the limits that are set, or None when none is."""
    present = [limit for limit in limits if limit is not None]
    return min(present) if present else None


def _rank(candidate: Candidate, tier: int, request: Request) -> tuple[int, float, bool, str]:
    """Sort key: tier, then soonest reset (free) or lowest price (paid), then a fresh account.

    Moving to a different account than the one just used breaks ties, so a job whose
    session hit its limit continues elsewhere when an equal choice exists.
    """
    detail = 0.0
    if tier == FREE:
        detail = candidate.resets_at.timestamp() if candidate.resets_at else float("inf")
    elif tier == PAID:
        detail = float(candidate.price_per_second)
    return (tier, detail, candidate.name == request.previous, candidate.name)
