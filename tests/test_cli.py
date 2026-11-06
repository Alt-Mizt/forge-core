"""The command line reads its options as documented and prints the tables it promises."""

import sys

import pytest

from forge import cli


def run_options(*arguments):
    return cli._parser().parse_args(["run", "train.py", *arguments])


@pytest.mark.parametrize(
    ("arguments", "gpu"),
    [
        ((), None),
        (("--gpu",), "any"),
        (("--gpu", "any"), "any"),
        (("--gpu", "a100-80gb"), "a100-80gb"),
    ],
)
def test_gpu_takes_a_name_any_or_nothing(arguments, gpu):
    assert run_options(*arguments).gpu == gpu


def forge(monkeypatch, capsys, tmp_path, *arguments):
    monkeypatch.setenv("FORGE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["forge", *arguments])
    status = cli.main()
    return status, capsys.readouterr()


def test_gpus_lists_each_gpu_once_with_where_it_is_and_its_price(monkeypatch, capsys, tmp_path):
    forge(monkeypatch, capsys, tmp_path, "add", "kaggle", "--credentials", "token")
    forge(monkeypatch, capsys, tmp_path, "add", "modal")
    status, printed = forge(monkeypatch, capsys, tmp_path, "gpus")
    assert status == 0
    rows = {line.split()[0]: line for line in printed.out.splitlines()[1:]}
    # The T4 is free on Kaggle; on Modal it costs $0.00053512 a second, $1.93 an hour.
    assert rows["T4"].split()[1:] == ["16", "GB", "kaggle,", "modal", "free,", "$1.93"]
    assert rows["A100-80GB"].split()[1:] == ["80", "GB", "modal", "$3.83"]


def test_a_gpu_no_account_offers_refuses_the_run(monkeypatch, capsys, tmp_path):
    forge(monkeypatch, capsys, tmp_path, "add", "local", "--device", "cpu")
    script = tmp_path / "train.py"
    script.write_text("")
    status, printed = forge(monkeypatch, capsys, tmp_path, "run", str(script), "--gpu", "H100")
    assert status == 1
    assert "No account offers H100: local offers cpu." in printed.err


def test_accounts_shows_when_a_one_off_credit_expires(monkeypatch, capsys, tmp_path):
    arguments = ("add", "modal", "--name", "grant", "--allowance", "100", "--expires", "2027-01-31")
    forge(monkeypatch, capsys, tmp_path, *arguments)
    _, printed = forge(monkeypatch, capsys, tmp_path, "accounts")
    assert "100.00 USD  expires 2027-01-31" in printed.out


def test_an_expiry_without_an_amount_refuses_on_the_command_line(monkeypatch, capsys, tmp_path):
    status, printed = forge(
        monkeypatch, capsys, tmp_path, "add", "modal", "--expires", "2027-01-31"
    )
    assert status == 1 and "Give its amount too" in printed.err


def test_plan_prints_the_route_and_starts_nothing(monkeypatch, capsys, tmp_path):
    for name in ("first", "second"):
        arguments = ("add", "local", "--name", name, "--device", "cpu", "--session-limit", "10")
        forge(monkeypatch, capsys, tmp_path, *arguments)
    script = tmp_path / "job" / "train.py"
    script.parent.mkdir()
    script.write_text("")
    hours = str(25 / 3600)
    status, printed = forge(monkeypatch, capsys, tmp_path, "plan", str(script), "--hours", hours)
    assert status == 0
    rows = [line.split()[:4] for line in printed.out.splitlines()[1:4]]
    assert rows == [
        ["1", "first", "cpu", "10"],
        ["2", "second", "cpu", "10"],
        ["3", "first", "cpu", "5"],
    ]
    assert "These 3 sessions cover the job" in printed.out
    assert list((tmp_path / "state" / "runs").iterdir()) == []
