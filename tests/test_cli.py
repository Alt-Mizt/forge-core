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
