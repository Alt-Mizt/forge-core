"""Holds come from the run journals, recorded sessions from the usage log, and they add up."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from forge import accounting, accounts, jobs, loop, runs
from forge.core.placement import Request
from forge.home import Home

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/tiny/train.py"
NOW = datetime(2026, 10, 7, 12, tzinfo=UTC)
HOUR = 3600


def home_with(tmp_path, *plugged):
    home = Home(tmp_path / "config", tmp_path / "state")
    home.prepare()
    accounts.save(home.accounts, list(plugged))
    return home


def start_session(home, account, seconds, hold=None):
    """A run whose first session has started on ``account`` and is not recorded yet."""
    name = f"run-{len(runs.all_journals(home.runs)) + 1}"
    loop.start(home, jobs.describe(SCRIPT, []), Request(), name)
    path = home.runs / name / "journal.json"
    journal = runs.read(path)
    journal["sessions"].append(
        {
            "number": 1,
            "account": account.name,
            "gpu": account.gpus[0].name,
            "handle": f"forge-{name}-1",
            "phase": runs.ACTIVE,
            "start": "2026-10-07T11:00:00Z",
            "seconds": seconds,
            "hold": hold
            if hold is not None
            else accounting.hold(account, account.gpus[0].name, seconds),
        }
    )
    runs.write(path, journal)
    return journal


def test_a_running_sessions_hold_comes_off_the_time_left_until_it_is_recorded(tmp_path):
    kaggle = accounts.new_account("kaggle")
    home = home_with(tmp_path, kaggle)
    journal = start_session(home, kaggle, 2 * HOUR)
    assert journal["sessions"][0]["hold"] == "2.000000000"
    assert accounting.remaining(home, [kaggle], NOW)["kaggle"].amount == Decimal(28)
    assert accounting.candidates(home, [kaggle], NOW)[0].busy

    journal["sessions"][0]["end"] = "2026-10-07T11:30:00Z"
    accounting.record_session(home, journal, kaggle)
    journal["sessions"][0]["phase"] = runs.RECORDED
    runs.write(home.runs / journal["run"] / "journal.json", journal)
    # Half an hour was used, so the other hour and a half comes back.
    assert accounting.remaining(home, [kaggle], NOW)["kaggle"].amount == Decimal("29.5")
    assert not accounting.candidates(home, [kaggle], NOW)[0].busy


def test_recorded_use_never_exceeds_what_was_held(tmp_path):
    kaggle = accounts.new_account("kaggle")
    home = home_with(tmp_path, kaggle)
    journal = start_session(home, kaggle, HOUR)
    journal["sessions"][0]["end"] = "2026-10-07T14:00:00Z"  # the provider ran long
    accounting.record_session(home, journal, kaggle)
    accounting.record_session(home, journal, kaggle)  # a second call changes nothing
    assert len(home.usage.read_text().splitlines()) == 1
    assert accounting.remaining(home, [kaggle], NOW)["kaggle"].amount == Decimal(28)


def test_modal_holds_the_collection_time_too_and_counts_dollars(tmp_path):
    modal = accounts.new_account("modal")
    # A T4 sandbox: $0.000164 for the GPU, plus 4 cores at $0.00003942 and 32 GiB at
    # $0.00000667, is $0.00053512 a second; (150 + 120) seconds of it is $0.1444824.
    assert accounting.hold(modal, "T4", 150) == "0.14448240"
    home = home_with(tmp_path, modal)
    start_session(home, modal, 150)
    assert accounting.remaining(home, [modal], NOW)["modal"].amount == Decimal("29.85551760")


def test_an_a100_session_holds_an_hour_at_the_a100_price_and_records_the_a100(tmp_path):
    modal = accounts.new_account("modal")
    # $0.000694 for the GPU plus $0.00037112 for the sandbox is $0.00106512 a second;
    # an hour plus the 120 s to collect results is 3720 s, so $3.9622464 is held.
    assert accounting.hold(modal, "A100-80GB", HOUR) == "3.96224640"
    home = home_with(tmp_path, modal)
    journal = start_session(home, modal, HOUR, hold=accounting.hold(modal, "A100-80GB", HOUR))
    journal["sessions"][0]["gpu"] = "A100-80GB"
    journal["sessions"][0]["end"] = "2026-10-07T11:30:00Z"
    accounting.record_session(home, journal, modal)
    line = json.loads(home.usage.read_text())
    # Half an hour, 1800 s, at $0.00106512.
    assert (line["gpu"], line["gpu_count"], line["amount"]) == ("A100-80GB", 1, "1.91721600")


def test_a_local_account_has_no_allowance_and_holds_nothing(tmp_path):
    local = accounts.new_account("local", device="cpu")
    home = home_with(tmp_path, local)
    assert accounting.hold(local, "cpu", 10) is None
    assert accounting.remaining(home, [local], NOW) == {}
    [candidate] = accounting.candidates(home, [local], NOW)
    assert candidate.local and candidate.unit is None
    assert [gpu.name for gpu in candidate.gpus] == ["cpu"]


def test_the_run_journal_records_the_job_declaration_and_the_request(tmp_path):
    home = home_with(tmp_path, accounts.new_account("modal"))
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "train.py").write_text("")
    (folder / "forge.toml").write_text('[job]\ngpu = "A100-80GB"\nmemory_gb = 40\n')
    job = jobs.describe(folder / "train.py", [])
    request = Request(gpu=job.declaration.gpu, memory_gb=job.declaration.memory_gb)
    loop.start(home, job, request, "declared")
    journal = runs.read(home.runs / "declared" / "journal.json")
    assert journal["job"]["declaration"] == {"gpu": "A100-80GB", "memory_gb": 40, "hours": None}
    assert journal["request"] == {"seconds": None, "gpu": "A100-80GB", "memory_gb": 40}
