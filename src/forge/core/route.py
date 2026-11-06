"""The route a job would take across the accounts: placement run forward, nothing started.

From the accounts as they stand, place a session, take its time off that account, and place
the next one, until the job's length is covered or no account can take the rest. Sessions
run one after another, so an account that allows one session at a time is free again for
the next. The route counts no resets along the way; a credit that has expired by the time a
session would start has nothing left for it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal

from forge.core.placement import Candidate, Placement, Request, choose, shortest
from forge.core.usage import amount_for


@dataclass(frozen=True)
class Route:
    """The sessions a job would run, in order, and what each account would have left.

    ``covered`` is True when the sessions add up to the whole job.
    """

    sessions: tuple[Placement, ...]
    left: tuple[Candidate, ...]
    covered: bool


def route(
    candidates: list[Candidate], request: Request, seconds: int | None, now: datetime
) -> Route:
    """Plan the sessions for a job of ``seconds``, starting at ``now``.

    With ``seconds`` None the job's length is unknown, so only the next session is planned.
    """
    left = tuple(candidates)
    sessions: list[Placement] = []
    needed = seconds
    clock = now
    previous = request.previous
    while needed is None or needed > 0:
        left = tuple(_expire(candidate, clock) for candidate in left)
        asked = replace(request, seconds=shortest(request.seconds, needed), previous=previous)
        placement = choose(list(left), asked)
        if placement is None:
            break
        sessions.append(placement)
        if needed is None:
            break
        used = placement.seconds if placement.seconds is not None else needed
        left = tuple(_spend(candidate, placement, used) for candidate in left)
        needed -= used
        clock += timedelta(seconds=used)
        previous = placement.account
    return Route(tuple(sessions), left, covered=needed is not None and needed <= 0)


def _expire(candidate: Candidate, clock: datetime) -> Candidate:
    """The account at ``clock``: a one-off credit that has expired by then has nothing left."""
    if candidate.expires is not None and candidate.expires <= clock:
        return replace(candidate, free=Decimal(0), paid=Decimal(0))
    return candidate


def _spend(candidate: Candidate, placement: Placement, seconds: int) -> Candidate:
    """The account after a session of ``seconds``, plus the provider's extra time, ran on it."""
    if candidate.name != placement.account or candidate.unit is None:
        return candidate
    gpu = next(gpu for gpu in candidate.gpus if gpu.name == placement.gpu)
    amount = amount_for(seconds + candidate.extra_seconds, candidate.unit, gpu.usd_per_second)
    if placement.paid:
        return replace(candidate, paid=max(candidate.paid - amount, Decimal(0)))
    return replace(candidate, free=max(candidate.free - amount, Decimal(0)))
