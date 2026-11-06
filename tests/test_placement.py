"""Hand-checked choices of where a job's next session runs, on which GPU, and for how long."""

from datetime import UTC, datetime
from decimal import Decimal

from forge.core.placement import Candidate, Gpu, Request, choose, mismatch, next_reset
from forge.core.quota import Unit

SOON = datetime(2026, 10, 10, tzinfo=UTC)
LATER = datetime(2026, 11, 1, tzinfo=UTC)
FREE_T4 = Gpu("T4", 16, 2, Decimal(0))
T4 = Gpu("T4", 16, 1, Decimal("0.0005"))
A100 = Gpu("A100-80GB", 80, 1, Decimal("0.001"))
CPU = Gpu("cpu", 0, 1, Decimal(0))
CUDA = Gpu("cuda", 0, 1, Decimal(0))


def candidate(name, **changes):
    """A Kaggle-like account: one free T4, two GPU-hours (7200 s) left, hour-long sessions."""
    values = {
        "name": name,
        "local": False,
        "gpus": (FREE_T4,),
        "usable": True,
        "busy": False,
        "session_limit": 3600,
        "unit": Unit.GPU_HOURS,
        "free": Decimal(2),
        "paid": Decimal(0),
        "extra_seconds": 0,
        "resets_at": LATER,
        "expires": None,
    }
    values.update(changes)
    return Candidate(**values)


def modal(name="modal", **changes):
    """A Modal-like account: dollars, a T4 and an A100, 120 s to collect results."""
    values = {"gpus": (T4, A100), "unit": Unit.USD, "free": Decimal(30), "extra_seconds": 120}
    return candidate(name, **(values | changes))


def local(name, gpu=CPU, **changes):
    """This machine: unlimited time, no session limit, one device."""
    values = {"local": True, "gpus": (gpu,), "unit": None, "free": Decimal(0)}
    return candidate(name, session_limit=None, resets_at=None, **(values | changes))


def test_a_local_gpu_comes_before_any_cloud_account():
    placement = choose([candidate("kaggle"), local("desk", CUDA)], Request(seconds=600))
    assert placement is not None
    assert (placement.account, placement.gpu, placement.seconds) == ("desk", "cuda", 600)
    assert not placement.paid


def test_a_local_cpu_account_is_skipped_when_the_job_needs_a_gpu():
    accounts = [local("desk"), candidate("kaggle")]
    placement = choose(accounts, Request(gpu="any"))
    assert placement is not None and (placement.account, placement.gpu) == ("kaggle", "T4")
    cpu_allowed = choose(accounts, Request())
    assert cpu_allowed is not None and (cpu_allowed.account, cpu_allowed.gpu) == ("desk", "cpu")


def test_an_a100_request_goes_to_the_only_account_that_has_one():
    # Kaggle resets sooner and would win on free time, but it has no A100.
    accounts = [candidate("kaggle", resets_at=SOON), modal()]
    placement = choose(accounts, Request(gpu="a100-80gb"))
    assert placement is not None
    assert (placement.account, placement.gpu, placement.paid) == ("modal", "A100-80GB", False)


def test_any_gpu_uses_the_cheapest_gpu_on_the_best_account():
    placement = choose([modal()], Request(gpu="any"))
    assert placement is not None and placement.gpu == "T4"


def test_free_dollars_last_fewer_seconds_on_a_dearer_gpu():
    # $1 buys 1000 s of A100 at $0.001 a second, less the 120 s held to collect results;
    # on the T4 at $0.0005 it buys 2000 s less 120.
    accounts = [modal(free=Decimal(1), session_limit=None)]
    a100 = choose(accounts, Request(gpu="A100-80GB"))
    t4 = choose(accounts, Request(gpu="T4"))
    assert a100 is not None and a100.seconds == 880
    assert t4 is not None and t4.seconds == 1880


def test_free_time_that_resets_soonest_is_used_first():
    accounts = [candidate("modal", resets_at=LATER), candidate("kaggle", resets_at=SOON)]
    placement = choose(accounts, Request())
    assert placement is not None and placement.account == "kaggle"


def test_paid_time_needs_a_paid_cap_and_goes_to_the_cheapest_chosen_gpu():
    empty = {"unit": Unit.USD, "free": Decimal(0), "paid": Decimal(1), "extra_seconds": 0}
    dear_a100 = Gpu("A100-80GB", 80, 1, Decimal("0.003"))
    accounts = [
        candidate("mixed", gpus=(T4, dear_a100), **empty),
        candidate("a100s", gpus=(Gpu("A100-80GB", 80, 1, Decimal("0.002")),), **empty),
        candidate("none", **(empty | {"paid": Decimal(0)})),
    ]
    # Any GPU: the T4 at $0.0005 is cheapest. An A100: $0.002 beats $0.003.
    cheapest = choose(accounts, Request())
    assert cheapest is not None
    assert (cheapest.account, cheapest.gpu, cheapest.paid) == ("mixed", "T4", True)
    a100 = choose(accounts, Request(gpu="A100-80GB"))
    assert a100 is not None and (a100.account, a100.seconds) == ("a100s", 500)
    assert choose([accounts[2]], Request()) is None


def test_failed_busy_and_exhausted_accounts_are_skipped():
    accounts = [
        candidate("failed", usable=False),
        candidate("busy", busy=True),
        candidate("empty", free=Decimal(0)),
    ]
    assert choose(accounts, Request()) is None


def test_session_length_is_the_smallest_of_request_limit_and_time_left():
    accounts = [candidate("kaggle", session_limit=3600, free=Decimal("0.5"))]
    long_request = choose(accounts, Request(seconds=7200))
    short_request = choose(accounts, Request(seconds=600))
    assert long_request is not None and long_request.seconds == 1800
    assert short_request is not None and short_request.seconds == 600


def test_on_names_the_account_even_when_another_ranks_higher():
    accounts = [candidate("kaggle", resets_at=SOON), candidate("modal")]
    placement = choose(accounts, Request(on="modal"))
    assert placement is not None and placement.account == "modal"


def test_an_equal_choice_moves_away_from_the_account_just_used():
    accounts = [local("first"), local("second")]
    placement = choose(accounts, Request(previous="first"))
    assert placement is not None and placement.account == "second"


def test_a_gpu_no_account_offers_is_refused_with_what_each_one_has():
    accounts = [candidate("kaggle"), modal(), local("desk")]
    reason = mismatch(accounts, Request(gpu="H100"))
    assert reason == (
        "No account offers H100: kaggle offers T4 (16 GB); modal offers T4 (16 GB), "
        "A100-80GB (80 GB); desk offers cpu. See them all with: forge gpus"
    )
    assert mismatch(accounts, Request(gpu="a100-80gb")) is None
    assert mismatch(accounts, Request(gpu="A100-80GB", on="kaggle")) is not None


def test_the_next_reset_is_the_soonest_one():
    assert next_reset([candidate("modal"), candidate("kaggle", resets_at=SOON)]) == SOON


def test_free_time_lost_first_is_spent_first():
    # Wednesday 7 October: Kaggle's week resets on Saturday the 10th, a grant expires on the
    # 20th, Modal's month resets on 1 November, and a credit with no expiry waits for last.
    kaggle = candidate("kaggle", resets_at=SOON)
    grant = modal("grant", resets_at=None, expires=datetime(2026, 10, 20, tzinfo=UTC))
    monthly = modal("modal", resets_at=LATER)
    forever = modal("forever", resets_at=None)
    order = []
    remaining = [forever, monthly, grant, kaggle]
    while (placement := choose(remaining, Request())) is not None:
        order.append(placement.account)
        remaining = [option for option in remaining if option.name != placement.account]
    assert order == ["kaggle", "grant", "modal", "forever"]
