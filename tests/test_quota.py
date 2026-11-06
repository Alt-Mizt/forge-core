"""Hand-checked cases for the free time left on an account."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from forge.core.quota import Allowance, Period, Unit, Use, remaining

SATURDAY = 5
KAGGLE = Allowance("kaggle", Decimal(30), Unit.GPU_HOURS, Period.WEEKLY, reset_weekday=SATURDAY)
MODAL = Allowance("modal", Decimal(30), Unit.USD, Period.MONTHLY)
WEDNESDAY_NOON = datetime(2026, 10, 7, 12, tzinfo=UTC)
NOTHING_HELD = Decimal(0)


def at(day: int, hour: int, month: int = 10, year: int = 2026) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def test_an_unused_weekly_allowance_is_all_left():
    left = remaining(KAGGLE, [], NOTHING_HELD, WEDNESDAY_NOON)
    assert left.amount == Decimal(30)
    assert left.resets_at == at(10, 0)  # the next Saturday


def test_weekly_allowances_reset_on_the_configured_day():
    # Saturday 3 Oct started the week, so Friday's session belongs to the week before.
    uses = [Use(ended=at(2, 18), amount=Decimal(10)), Use(ended=at(5, 13), amount=Decimal(4))]
    assert remaining(KAGGLE, uses, NOTHING_HELD, WEDNESDAY_NOON).amount == Decimal(26)


def test_a_session_that_ended_after_the_reset_counts_in_full():
    # Friday 22:00 to Saturday 04:00 is recorded as six hours, all in the new week.
    uses = [Use(ended=at(3, 4), amount=Decimal(6))]
    assert remaining(KAGGLE, uses, NOTHING_HELD, WEDNESDAY_NOON).amount == Decimal(24)


def test_held_time_comes_off_what_is_left():
    uses = [Use(ended=at(6, 10), amount=Decimal(3))]
    assert remaining(KAGGLE, uses, Decimal(9), WEDNESDAY_NOON).amount == Decimal(18)


def test_a_dollar_allowance_resets_monthly():
    uses = [
        Use(ended=at(5, 15), amount=Decimal("12.60")),
        Use(ended=at(30, 11, month=9), amount=Decimal(4)),  # September's belongs to last month
    ]
    left = remaining(MODAL, uses, NOTHING_HELD, WEDNESDAY_NOON)
    assert left.amount == Decimal("17.40")
    assert left.resets_at == at(1, 0, month=11)


def test_an_overspent_allowance_shows_zero_not_a_negative_amount():
    uses = [Use(ended=at(5, 0), amount=Decimal(48))]
    assert remaining(KAGGLE, uses, NOTHING_HELD, WEDNESDAY_NOON).amount == Decimal(0)


def test_monthly_resets_roll_over_into_the_next_year():
    left = remaining(MODAL, [], NOTHING_HELD, at(15, 12, month=12))
    assert left.resets_at == at(1, 0, month=1, year=2027)


def test_naive_times_are_rejected_with_guidance():
    with pytest.raises(ValueError, match="has no time zone"):
        remaining(KAGGLE, [], NOTHING_HELD, datetime(2026, 10, 7, 12))


GRANT = Allowance(
    "grant", Decimal(100), Unit.USD, Period.ONCE, expires=datetime(2027, 1, 31, tzinfo=UTC)
)


def test_a_one_off_credit_counts_every_use_and_never_resets():
    # Uses from last month and this one both count: a one-off credit has no period.
    uses = [
        Use(ended=at(30, 11, month=9), amount=Decimal(40)),
        Use(ended=at(5, 9), amount=Decimal(25)),
    ]
    left = remaining(GRANT, uses, Decimal(5), WEDNESDAY_NOON)
    assert left.amount == Decimal(30)
    assert left.resets_at is None
    assert left.expires == datetime(2027, 1, 31, tzinfo=UTC)


def test_an_expired_credit_has_nothing_left():
    assert remaining(GRANT, [], NOTHING_HELD, at(30, 23, month=1, year=2027)).amount == 100
    assert remaining(GRANT, [], NOTHING_HELD, at(31, 0, month=1, year=2027)).amount == 0
