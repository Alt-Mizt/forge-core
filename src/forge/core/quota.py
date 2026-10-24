"""Free time left on an account.

A provider gives free time as an allowance that starts again on a schedule: 30 GPU-hours
a week, $30 of credit a month. What is left is that allowance, less the sessions the usage
log has recorded since the period began, less the time held for sessions still running.

Nothing here reads a clock. The current time is passed in, so every answer can be checked
by hand.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

DAYS_PER_WEEK = 7
DECEMBER = 12


class Period(Enum):
    """How often an allowance starts again. Resets happen at 00:00 UTC."""

    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Unit(Enum):
    """What an allowance is measured in."""

    GPU_HOURS = "gpu-hours"
    USD = "usd"


@dataclass(frozen=True)
class Allowance:
    """What an account may use per period.

    Attributes:
        account: The account's name, as its sessions carry it in the usage log.
        amount: How much may be used in each period.
        unit: Whether the amount is GPU-hours or dollars of credit.
        period: How often the allowance starts again.
        reset_weekday: For weekly allowances, the day it starts again (Monday is 0).
            Monthly allowances always start again on the 1st.
    """

    account: str
    amount: Decimal
    unit: Unit
    period: Period
    reset_weekday: int = 0


@dataclass(frozen=True)
class Use:
    """One recorded session's share of an allowance: when it ended and how much it took."""

    ended: datetime
    amount: Decimal


@dataclass(frozen=True)
class Remaining:
    """What is left of one allowance, and when it starts again."""

    account: str
    amount: Decimal
    unit: Unit
    resets_at: datetime


def remaining(allowance: Allowance, uses: Iterable[Use], held: Decimal, now: datetime) -> Remaining:
    """Work out what is left of ``allowance`` at ``now``.

    A use counts in the period it ended in. A session that spans a reset counts in full in
    the new period: the log has no per-period meter, so splitting its use would be a guess.
    ``held`` is the time already promised to sessions that are still running. An overspent
    allowance shows zero, never a negative amount.

    Raises:
        ValueError: If ``now`` is naive, because local time would shift the resets.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(
            f"The time {now.isoformat()} has no time zone. Use UTC, for example datetime.now(UTC)."
        )
    start = current_period_start(allowance, now)
    used = sum((use.amount for use in uses if use.ended > start), Decimal(0))
    return Remaining(
        account=allowance.account,
        amount=max(allowance.amount - used - held, Decimal(0)),
        unit=allowance.unit,
        resets_at=next_reset(allowance, now),
    )


def current_period_start(allowance: Allowance, now: datetime) -> datetime:
    """The most recent reset at or before ``now``."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if allowance.period is Period.MONTHLY:
        return midnight.replace(day=1)
    days_since_reset = (now.weekday() - allowance.reset_weekday) % DAYS_PER_WEEK
    return midnight - timedelta(days=days_since_reset)


def next_reset(allowance: Allowance, now: datetime) -> datetime:
    """The first reset after ``now``."""
    start = current_period_start(allowance, now)
    if allowance.period is Period.WEEKLY:
        return start + timedelta(days=DAYS_PER_WEEK)
    if start.month == DECEMBER:
        return start.replace(year=start.year + 1, month=1)
    return start.replace(month=start.month + 1)
