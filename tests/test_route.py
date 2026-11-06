"""Hand-checked routes: placement run forward over the accounts, with nothing started."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from forge.core.placement import Candidate, Gpu, Request
from forge.core.quota import Unit
from forge.core.route import route

NOW = datetime(2026, 10, 7, 12, tzinfo=UTC)  # a Wednesday
HOUR = 3600
CPU = Gpu("cpu", 0, 1, Decimal(0))
KAGGLE_T4 = Gpu("T4", 16, 2, Decimal(0))
MODAL_T4 = Gpu("T4", 16, 1, Decimal("0.00053512"))


def account(name, **values):
    defaults = {
        "name": name,
        "local": False,
        "gpus": (KAGGLE_T4,),
        "usable": True,
        "busy": False,
        "session_limit": None,
        "unit": None,
        "free": Decimal(0),
        "paid": Decimal(0),
        "extra_seconds": 0,
        "resets_at": None,
        "expires": None,
    }
    return Candidate(**(defaults | values))


def local(name):
    return account(name, local=True, gpus=(CPU,), session_limit=10)


def test_two_ten_second_accounts_take_turns_through_a_25_second_job():
    planned = route([local("first"), local("second")], Request(), 25, NOW)
    sessions = [(session.account, session.seconds) for session in planned.sessions]
    assert sessions == [("first", 10), ("second", 10), ("first", 5)]
    assert planned.covered


def test_a_ten_hour_job_spends_kaggle_first_then_modal():
    # Kaggle has 5 of its 30 GPU-hours left and resets on Saturday; Modal has $30 and
    # resets on 1 November. Kaggle goes first and is spent in one 5-hour session; Modal
    # takes the other 5 hours, 18000 s, plus 120 s to collect results: 18120 s at
    # $0.00053512 is $9.6963744, leaving $20.3036256.
    kaggle = account(
        "kaggle",
        session_limit=9 * HOUR,
        unit=Unit.GPU_HOURS,
        free=Decimal(5),
        resets_at=datetime(2026, 10, 10, tzinfo=UTC),
    )
    modal = account(
        "modal",
        gpus=(MODAL_T4,),
        session_limit=24 * HOUR,
        unit=Unit.USD,
        free=Decimal(30),
        extra_seconds=120,
        resets_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    planned = route([modal, kaggle], Request(gpu="any"), 10 * HOUR, NOW)
    sessions = [(session.account, session.gpu, session.seconds) for session in planned.sessions]
    assert sessions == [("kaggle", "T4", 5 * HOUR), ("modal", "T4", 5 * HOUR)]
    assert planned.covered
    left = {candidate.name: candidate.free for candidate in planned.left}
    assert left == {"kaggle": Decimal(0), "modal": Decimal("20.3036256")}


def test_a_credit_that_expires_along_the_way_is_not_planned_on_after_it_expires():
    grant = account(
        "grant",
        session_limit=HOUR,
        unit=Unit.GPU_HOURS,
        free=Decimal(10),
        expires=NOW + timedelta(hours=1, minutes=30),
    )
    planned = route([grant], Request(), 3 * HOUR, NOW)
    assert [session.seconds for session in planned.sessions] == [HOUR, HOUR]
    assert not planned.covered


def test_without_a_job_length_only_the_next_session_is_planned():
    planned = route([local("first"), local("second")], Request(), None, NOW)
    assert [session.account for session in planned.sessions] == ["first"]
    assert not planned.covered
