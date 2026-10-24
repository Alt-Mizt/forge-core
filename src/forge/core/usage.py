"""Convert between session seconds and the unit an account's allowance is counted in."""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from forge.core.quota import Unit

SECONDS_PER_HOUR = Decimal(3600)
# GPU-hours are kept to nine decimal places, finer than any provider meters.
RESOLUTION = Decimal("0.000000001")


def amount_for(seconds: int, unit: Unit, gpu_count: int, usd_per_second: Decimal) -> Decimal:
    """What ``seconds`` of a session costs in ``unit``, rounded up so it is never understated.

    GPU-hours count every allocated GPU. Dollars use the provider's price per second.
    """
    if unit is Unit.GPU_HOURS:
        hours = Decimal(seconds * gpu_count) / SECONDS_PER_HOUR
        return hours.quantize(RESOLUTION, rounding=ROUND_CEILING)
    return usd_per_second * Decimal(seconds)


def seconds_for(amount: Decimal, unit: Unit, gpu_count: int, usd_per_second: Decimal) -> int:
    """How many whole seconds of a session ``amount`` pays for, rounded down."""
    if unit is Unit.GPU_HOURS:
        seconds = amount * SECONDS_PER_HOUR / gpu_count
    elif usd_per_second > 0:
        seconds = amount / usd_per_second
    else:
        return 0
    return int(seconds.to_integral_value(rounding=ROUND_FLOOR))
