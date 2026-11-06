"""Choose which account a job's next session runs on, on which GPU, and for how long.

The caller describes each account as a ``Candidate``; nothing here reads a file or a clock.
The order of preference is: a local machine that fits the job, then free time, then paid
time on accounts that allow it, cheapest first. Free time is spent first where it is lost
first: recurring allowances by their next reset, one-off credits by their expiry, and a
credit that never expires last. On each account the session uses the cheapest GPU that
fits the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from forge.core.quota import Unit
from forge.core.usage import seconds_for

LOCAL, FREE, PAID = 0, 1, 2
# A session shorter than this cannot start a job, load a snapshot and save again.
MINIMUM_SECONDS = 1
# The GPU request that takes any GPU, and the name of a machine's CPU when it has none.
ANY, CPU = "any", "cpu"


@dataclass(frozen=True)
class Gpu:
    """One kind of GPU an account offers.

    Attributes:
        name: The provider's name for it, such as ``T4`` or ``A100-80GB``; on this machine,
            the torch device: ``cuda``, ``mps``, or ``cpu`` when there is no GPU.
        memory_gb: The memory of one GPU; 0 when forge does not know it.
        count: How many of them one session gets.
        usd_per_second: The most a session on it costs per second; 0 when it is free.
    """

    name: str
    memory_gb: int
    count: int
    usd_per_second: Decimal


@dataclass(frozen=True)
class Candidate:
    """One account as placement sees it.

    Attributes:
        name: The account name.
        local: Whether it is this machine.
        gpus: The GPUs it offers, cheapest first.
        usable: False when its credentials failed their last check.
        busy: True when it already runs a session and allows only one at a time.
        session_limit: Longest session in seconds, or None for no limit.
        unit: What its allowance is counted in, or None when its time is unlimited.
        free: Free amount left, in ``unit``.
        paid: Paid amount left under the account's paid cap, in ``unit``.
        extra_seconds: Seconds a session stays billable after its limit, to hand back results.
        resets_at: When the free allowance starts again, if it has one.
        expires: When a one-off credit is gone, if it has an expiry.
    """

    name: str
    local: bool
    gpus: tuple[Gpu, ...]
    usable: bool
    busy: bool
    session_limit: int | None
    unit: Unit | None
    free: Decimal
    paid: Decimal
    extra_seconds: int
    resets_at: datetime | None
    expires: datetime | None


@dataclass(frozen=True)
class Request:
    """What the job asks for: a session length, a GPU, an account, and where it last ran.

    ``gpu`` is a GPU's name, ``any`` for any GPU, or None when the CPU will do.
    """

    seconds: int | None = None
    gpu: str | None = None
    on: str | None = None
    previous: str | None = None


@dataclass(frozen=True)
class Placement:
    """Where the next session runs, on which GPU, for how long, and whether it uses paid time.

    ``seconds`` is None only for a local account with no limit: the job runs to the end.
    """

    account: str
    gpu: str
    seconds: int | None
    paid: bool


def choose(candidates: list[Candidate], request: Request) -> Placement | None:
    """Return the best placement for the next session, or None when no account can take it."""
    options: list[tuple[tuple[int, float, bool, str], Placement]] = []
    for candidate in candidates:
        gpu = gpu_for(candidate, request)
        if gpu is None or not _eligible(candidate, request):
            continue
        tier, seconds = _time_left(candidate, gpu)
        seconds = shortest(seconds, candidate.session_limit, request.seconds)
        if seconds is not None and seconds < MINIMUM_SECONDS:
            continue
        placement = Placement(candidate.name, gpu.name, seconds, paid=tier == PAID)
        options.append((_rank(candidate, tier, gpu, request), placement))
    if not options:
        return None
    return min(options, key=lambda option: option[0])[1]


def gpu_for(candidate: Candidate, request: Request) -> Gpu | None:
    """The cheapest GPU on the account that fits the request, or None when none does."""
    for gpu in candidate.gpus:
        if _fits(gpu, request):
            return gpu
    return None


def mismatch(candidates: list[Candidate], request: Request) -> str | None:
    """Why no account could ever take the request, whatever time it has left; else None."""
    named = [candidate for candidate in candidates if request.on in (None, candidate.name)]
    if not named or any(gpu_for(candidate, request) for candidate in named):
        return None
    offers = "; ".join(f"{candidate.name} offers {_offers(candidate)}" for candidate in named)
    return f"No account offers {_wanted(request)}: {offers}. See them all with: forge gpus"


def next_reset(candidates: list[Candidate]) -> datetime | None:
    """The soonest time any account's free allowance starts again."""
    resets = [candidate.resets_at for candidate in candidates if candidate.resets_at]
    return min(resets) if resets else None


def seconds_left(candidate: Candidate, amount: Decimal, gpu: Gpu) -> int:
    """Whole seconds of session on ``gpu`` that ``amount`` pays for, after the extra time."""
    if candidate.unit is None:
        return 0
    total = seconds_for(amount, candidate.unit, gpu.usd_per_second)
    return max(total - candidate.extra_seconds, 0)


def _fits(gpu: Gpu, request: Request) -> bool:
    """Whether one GPU is what the request asks for."""
    if request.gpu is None:
        return True
    if request.gpu.lower() == ANY:
        return gpu.name != CPU
    return gpu.name.lower() == request.gpu.lower()


def _eligible(candidate: Candidate, request: Request) -> bool:
    """Whether the account may take this session at all, before comparing time left."""
    if request.on is not None and candidate.name != request.on:
        return False
    return candidate.usable and not candidate.busy


def _time_left(candidate: Candidate, gpu: Gpu) -> tuple[int, int | None]:
    """Which kind of time the account would use next, and how many seconds of it are left."""
    tier = LOCAL if candidate.local else FREE
    if candidate.unit is None:
        return tier, None
    free = seconds_left(candidate, candidate.free, gpu)
    if candidate.local or free >= MINIMUM_SECONDS:
        return tier, free
    return PAID, seconds_left(candidate, candidate.paid, gpu)


def shortest(*limits: int | None) -> int | None:
    """The smallest of the limits that are set, or None when none is."""
    present = [limit for limit in limits if limit is not None]
    return min(present) if present else None


def _rank(
    candidate: Candidate, tier: int, gpu: Gpu, request: Request
) -> tuple[int, float, bool, str]:
    """Sort key: tier, then the free time lost first or the GPU's price, then a fresh account.

    Moving to a different account than the one just used breaks ties, so a job whose
    session hit its limit continues elsewhere when an equal choice exists.
    """
    detail = 0.0
    if tier == FREE:
        lost = candidate.resets_at or candidate.expires
        detail = lost.timestamp() if lost else float("inf")
    elif tier == PAID:
        detail = float(gpu.usd_per_second)
    return (tier, detail, candidate.name == request.previous, candidate.name)


def _offers(candidate: Candidate) -> str:
    """The GPUs an account offers, in words."""
    names = [
        f"{gpu.name} ({gpu.memory_gb} GB)" if gpu.memory_gb else gpu.name for gpu in candidate.gpus
    ]
    return ", ".join(names) or "nothing"


def _wanted(request: Request) -> str:
    """What the request asks for, in words."""
    if request.gpu is None or request.gpu.lower() == ANY:
        return "a GPU"
    return request.gpu
