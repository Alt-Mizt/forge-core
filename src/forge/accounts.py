"""The accounts plugged into forge, kept in ``accounts.toml`` in forge's config folder.

An account stores a reference to its credentials (a token file path or a Modal profile
name), never the secret itself. Each provider brings defaults for its free allowance and
its session limit, which ``forge add`` options can override.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from forge import files
from forge.core.quota import Allowance, Period, Unit

NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
PROVIDERS = ("kaggle", "modal", "local")
HOURS = 3600
SATURDAY = 5


@dataclass(frozen=True)
class Facts:
    """What forge assumes about a provider until an account says otherwise."""

    unit: Unit | None
    allowance: Decimal | None
    period: Period | None
    reset_weekday: int
    session_limit: int | None
    gpu: str
    gpu_count: int
    usd_per_second: Decimal
    one_session_at_a_time: bool
    # Seconds a session may stay billable after the job's own limit, to hand back results.
    extra_seconds: int = 0


FACTS = {
    # Kaggle gives about 30 GPU-hours a week. Its T4 machine has two T4s, but Kaggle
    # counts a session's time once.
    "kaggle": Facts(
        unit=Unit.GPU_HOURS,
        allowance=Decimal(30),
        period=Period.WEEKLY,
        reset_weekday=SATURDAY,
        session_limit=9 * HOURS,
        gpu="T4",
        gpu_count=1,
        usd_per_second=Decimal(0),
        one_session_at_a_time=True,
    ),
    # Modal gives $30 of credit a month. $0.001 a second bounds a T4 sandbox from above.
    "modal": Facts(
        unit=Unit.USD,
        allowance=Decimal(30),
        period=Period.MONTHLY,
        reset_weekday=0,
        session_limit=24 * HOURS,
        gpu="T4",
        gpu_count=1,
        usd_per_second=Decimal("0.001"),
        one_session_at_a_time=False,
        extra_seconds=120,
    ),
    # This machine: no allowance and no session limit unless the account sets one.
    "local": Facts(
        unit=None,
        allowance=None,
        period=None,
        reset_weekday=0,
        session_limit=None,
        gpu="local",
        gpu_count=1,
        usd_per_second=Decimal(0),
        one_session_at_a_time=True,
    ),
}


@dataclass(frozen=True)
class Account:
    """One plugged-in account.

    Attributes:
        name: Unique short name; also the name its sessions carry in the usage log.
        provider: ``kaggle``, ``modal`` or ``local``.
        credentials: Path to a Kaggle token file, or a Modal profile name; None for local.
        allowance: Free amount per period, in the provider's unit; None means unlimited.
        paid_cap: How much paid time per period forge may use. Zero means free time only.
        session_limit: Longest single session in seconds; None means no limit.
        device: For a local account, ``cuda``, ``mps`` or ``cpu``.
    """

    name: str
    provider: str
    credentials: str | None
    allowance: Decimal | None
    paid_cap: Decimal
    session_limit: int | None
    device: str | None = None

    @property
    def facts(self) -> Facts:
        """The provider defaults this account builds on."""
        return FACTS[self.provider]

    @property
    def unit(self) -> Unit | None:
        """The unit the allowance is counted in, or None for an unlimited account."""
        return self.facts.unit if self.allowance is not None else None

    def limit(self) -> Allowance | None:
        """The most the account may use per period: its free allowance plus its paid cap."""
        facts = self.facts
        if self.allowance is None or facts.unit is None or facts.period is None:
            return None
        total = self.allowance + self.paid_cap
        return Allowance(self.name, total, facts.unit, facts.period, facts.reset_weekday)


def new_account(
    provider: str,
    *,
    name: str | None = None,
    credentials: str | None = None,
    allowance: Decimal | None = None,
    paid_cap: Decimal | None = None,
    session_limit: int | None = None,
    device: str | None = None,
) -> Account:
    """Build an account from the provider's defaults, with any value given here instead."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Use one of: {', '.join(PROVIDERS)}.")
    name = name or provider
    if NAME.fullmatch(name) is None:
        raise ValueError("Account names are short lowercase words, like kaggle or lab-gpu.")
    facts = FACTS[provider]
    if paid_cap is not None and paid_cap < 0:
        raise ValueError("A paid cap cannot be negative.")
    return Account(
        name=name,
        provider=provider,
        credentials=credentials,
        allowance=allowance if allowance is not None else facts.allowance,
        paid_cap=paid_cap if paid_cap is not None else Decimal(0),
        session_limit=session_limit if session_limit is not None else facts.session_limit,
        device=device,
    )


def load(path: Path) -> list[Account]:
    """Read every account from ``accounts.toml``; a missing file means no accounts yet."""
    if not path.exists():
        return []
    tables = tomllib.loads(path.read_text()).get("account", {})
    return [_account(name, table) for name, table in tables.items()]


def save(path: Path, accounts: list[Account]) -> None:
    """Write the account list, replacing the file in one step."""
    lines = []
    for account in accounts:
        lines.append(f"[account.{account.name}]")
        for key, value in _fields(account).items():
            lines.append(f"{key} = {value}")
        lines.append("")
    files.write_atomically(path, "\n".join(lines).encode())


def find(accounts: list[Account], name: str) -> Account:
    """Return the account called ``name``, or say which names exist."""
    for account in accounts:
        if account.name == name:
            return account
    known = ", ".join(account.name for account in accounts) or "none yet"
    raise ValueError(f"No account named {name!r}. Accounts: {known}.")


def _fields(account: Account) -> dict[str, str]:
    """The TOML lines for one account, with numbers stored as exact strings."""
    fields = {"provider": account.provider, "paid_cap": str(account.paid_cap)}
    if account.credentials is not None:
        fields["credentials"] = account.credentials
    if account.allowance is not None:
        fields["allowance"] = str(account.allowance)
    if account.device is not None:
        fields["device"] = account.device
    quoted = {key: _quote(value) for key, value in fields.items()}
    if account.session_limit is not None:
        quoted["session_limit"] = str(account.session_limit)
    return quoted


def _quote(value: str) -> str:
    """A TOML basic string."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _account(name: str, table: dict[str, object]) -> Account:
    """Build one account from its TOML table, refusing unknown providers or names."""
    if NAME.fullmatch(name) is None or table.get("provider") not in PROVIDERS:
        raise ValueError(f"accounts.toml has an invalid account {name!r}.")
    allowance = table.get("allowance")
    session_limit = table.get("session_limit")
    return Account(
        name=name,
        provider=str(table["provider"]),
        credentials=str(table["credentials"]) if "credentials" in table else None,
        allowance=Decimal(str(allowance)) if allowance is not None else None,
        paid_cap=Decimal(str(table.get("paid_cap", "0"))),
        session_limit=int(str(session_limit)) if session_limit is not None else None,
        device=str(table["device"]) if "device" in table else None,
    )
