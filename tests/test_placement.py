"""Hand-checked choices of where a job's next session runs, and for how long."""

from datetime import UTC, datetime
from decimal import Decimal

from forge.core.placement import Candidate, Request, choose, next_reset

SOON = datetime(2026, 10, 10, tzinfo=UTC)
LATER = datetime(2026, 11, 1, tzinfo=UTC)


def candidate(name, **changes):
    values = {
        "name": name,
        "local": False,
        "has_gpu": True,
        "usable": True,
        "busy": False,
        "session_limit": 3600,
        "free_seconds": 7200,
        "paid_seconds": 0,
        "price_per_second": Decimal(0),
        "resets_at": LATER,
    }
    values.update(changes)
    return Candidate(**values)


def test_a_local_gpu_comes_before_any_cloud_account():
    local = candidate("desk", local=True, free_seconds=None, session_limit=None, resets_at=None)
    placement = choose([candidate("kaggle"), local], Request(seconds=600))
    assert placement is not None
    assert (placement.account, placement.seconds, placement.paid) == ("desk", 600, False)


def test_a_local_account_without_a_gpu_is_skipped_when_the_job_needs_one():
    local = candidate("desk", local=True, has_gpu=False, free_seconds=None, resets_at=None)
    placement = choose([local, candidate("kaggle")], Request(needs_gpu=True))
    assert placement is not None and placement.account == "kaggle"


def test_free_time_that_resets_soonest_is_used_first():
    accounts = [candidate("modal", resets_at=LATER), candidate("kaggle", resets_at=SOON)]
    placement = choose(accounts, Request())
    assert placement is not None and placement.account == "kaggle"


def test_paid_time_needs_a_paid_cap_and_goes_to_the_cheapest():
    empty = {"free_seconds": 0}
    accounts = [
        candidate("costly", paid_seconds=600, price_per_second=Decimal("0.002"), **empty),
        candidate("cheap", paid_seconds=600, price_per_second=Decimal("0.001"), **empty),
        candidate("none", paid_seconds=0, **empty),
    ]
    placement = choose(accounts, Request())
    assert placement is not None
    assert (placement.account, placement.paid) == ("cheap", True)
    assert choose([candidate("none", paid_seconds=0, **empty)], Request()) is None


def test_failed_busy_and_exhausted_accounts_are_skipped():
    accounts = [
        candidate("failed", usable=False),
        candidate("busy", busy=True),
        candidate("empty", free_seconds=0),
    ]
    assert choose(accounts, Request()) is None


def test_session_length_is_the_smallest_of_request_limit_and_time_left():
    accounts = [candidate("kaggle", session_limit=3600, free_seconds=1800)]
    long_request = choose(accounts, Request(seconds=7200))
    short_request = choose(accounts, Request(seconds=600))
    assert long_request is not None and long_request.seconds == 1800
    assert short_request is not None and short_request.seconds == 600


def test_on_names_the_account_even_when_another_ranks_higher():
    accounts = [candidate("kaggle", resets_at=SOON), candidate("modal")]
    placement = choose(accounts, Request(on="modal"))
    assert placement is not None and placement.account == "modal"


def test_an_equal_choice_moves_away_from_the_account_just_used():
    accounts = [candidate("first", local=True), candidate("second", local=True)]
    placement = choose(accounts, Request(previous="first"))
    assert placement is not None and placement.account == "second"


def test_the_next_reset_is_the_soonest_one():
    assert next_reset([candidate("modal"), candidate("kaggle", resets_at=SOON)]) == SOON
