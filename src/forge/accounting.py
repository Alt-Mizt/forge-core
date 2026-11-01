"""Connect accounts to the usage log and the run journals: time left, holds, recorded sessions.

Before a session starts, its whole possible length is held in its run's journal, so two
runs cannot plan on the same free time. When it ends, the time it took goes into the usage
log, and the hold ends with the session.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from forge import ledger, runs
from forge.accounts import Account
from forge.core.placement import Candidate
from forge.core.quota import Remaining, Unit, Use
from forge.core.quota import remaining as time_left
from forge.core.usage import amount_for, seconds_for
from forge.home import Home


def remaining(home: Home, accounts: list[Account], now: datetime) -> dict[str, Remaining]:
    """What is left of each account's allowance after recorded use and outstanding holds."""
    entries = ledger.read(home.usage)
    held = holds(home)
    left: dict[str, Remaining] = {}
    for account in accounts:
        allowance = account.limit()
        if allowance is None:
            continue
        uses = [Use(entry.end, entry.amount) for entry in entries if entry.account == account.name]
        left[account.name] = time_left(allowance, uses, held.get(account.name, Decimal(0)), now)
    return left


def candidates(home: Home, accounts: list[Account], now: datetime) -> list[Candidate]:
    """Describe every account for placement: health, whether busy, and time left."""
    left = remaining(home, accounts, now)
    failed = _failed_checks(home)
    busy = {session["account"] for session in _active_sessions(home)}
    return [_candidate(account, left.get(account.name), failed, busy) for account in accounts]


def hold(account: Account, seconds: int | None) -> str | None:
    """The amount a session of ``seconds`` may use at most, as text for the run's journal.

    None when the account has no allowance, or when the session has no time limit.
    """
    if account.limit() is None or seconds is None:
        return None
    return str(_amount(account, seconds + account.facts.extra_seconds))


def holds(home: Home) -> dict[str, Decimal]:
    """The time held per account by sessions that have not been recorded yet."""
    held: dict[str, Decimal] = {}
    for session in _active_sessions(home):
        if session.get("hold"):
            name = session["account"]
            held[name] = held.get(name, Decimal(0)) + Decimal(session["hold"])
    return held


def record_session(home: Home, journal: dict[str, Any], account: Account) -> None:
    """Write the run's last session to the usage log, unless it is there already.

    The usage is the session's wall-clock time, never more than what was held.
    """
    session = journal["sessions"][-1]
    if any(entry.session_id == session["handle"] for entry in ledger.read(home.usage)):
        return
    ledger.append(home.usage, _entry(journal, session, account))


def now() -> datetime:
    """The current time, for callers that should not read the clock themselves."""
    return datetime.now(UTC)


def _entry(journal: dict[str, Any], session: dict[str, Any], account: Account) -> dict[str, Any]:
    """One usage log line for an ended session."""
    elapsed = (ledger.timestamp(session["end"]) - ledger.timestamp(session["start"])).seconds
    amount = _amount(account, max(elapsed, 1))
    if session.get("hold"):
        amount = min(amount, Decimal(session["hold"]))
    return {
        "session_id": session["handle"],
        "account": account.name,
        "gpu": account.device or account.facts.gpu,
        "gpu_count": account.facts.gpu_count,
        "start": session["start"],
        "end": session["end"],
        "unit": (account.unit or Unit.GPU_HOURS).value,
        "amount": str(amount),
        "run": journal["run"],
        "job": Path(journal["job"]["folder"]).name or "job",
        "job_digest": journal["job"]["digest"][:12],
    }


def _amount(account: Account, seconds: int) -> Decimal:
    """What ``seconds`` cost on this account; local time is counted in hours of the device."""
    facts = account.facts
    return amount_for(
        seconds, account.unit or Unit.GPU_HOURS, facts.gpu_count, facts.usd_per_second
    )


def _candidate(
    account: Account, left: Remaining | None, failed: set[str], busy: set[str]
) -> Candidate:
    """Placement's view of one account."""
    facts = account.facts
    free_seconds, paid_seconds = None, 0
    if left is not None and account.unit is not None:
        free = max(left.amount - account.paid_cap, Decimal(0))
        paid = min(left.amount, account.paid_cap)
        free_seconds = _seconds(account, free)
        paid_seconds = _seconds(account, paid)
    return Candidate(
        name=account.name,
        local=account.provider == "local",
        has_gpu=account.provider != "local" or account.device not in (None, "cpu"),
        usable=account.name not in failed,
        busy=facts.one_session_at_a_time and account.name in busy,
        session_limit=account.session_limit,
        free_seconds=free_seconds,
        paid_seconds=paid_seconds,
        price_per_second=facts.usd_per_second,
        resets_at=left.resets_at if left is not None else None,
    )


def _seconds(account: Account, amount: Decimal) -> int:
    """Whole seconds of session that ``amount`` pays for, after the provider's extra time."""
    facts = account.facts
    unit = account.unit or Unit.GPU_HOURS
    total = seconds_for(amount, unit, facts.gpu_count, facts.usd_per_second)
    return max(total - facts.extra_seconds, 0)


def _failed_checks(home: Home) -> set[str]:
    """Accounts whose credentials failed their last ``forge check``."""
    if not home.checks.exists():
        return set()
    results = json.loads(home.checks.read_text())
    return {name for name, result in results.items() if not result["ok"]}


def _active_sessions(home: Home) -> list[dict[str, Any]]:
    """Every session, across all runs, that has started and is not recorded yet."""
    sessions = []
    for journal in runs.all_journals(home.runs):
        if (session := runs.active(journal)) is not None:
            sessions.append(session)
    return sessions
