"""forge runs your training job on whatever GPU time your accounts have.

Plug in accounts with "forge add", then "forge run train.py". When a session ends or an
account's free time runs out, the job pauses at a snapshot and continues on another account.
By default forge never spends money.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from forge import accounting, accounts, control, jobs, loop, runs
from forge.core.placement import ANY, Gpu, Request
from forge.core.quota import Remaining, Unit
from forge.home import Home, locate
from forge.providers import for_account

# Errors a command turns into a refusal instead of a traceback.
FAILURES = (OSError, InvalidOperation, KeyError, TypeError, RuntimeError)
DETACHED = 130
HOUR = 3600


def main() -> int:
    """Parse the command line, run the command and return its exit status."""
    arguments, job_arguments = _split(sys.argv[1:])
    options = _parser().parse_args(arguments)
    options.args = job_arguments
    home = locate()
    home.prepare()
    try:
        return COMMANDS[options.command](home, options)
    except loop.Detached as detached:
        sys.stdout.write(
            f"\nDetached; the session keeps running. Follow it again with: "
            f"forge resume {detached}\n"
        )
        return DETACHED
    except ValueError as error:
        sys.stderr.write(f"Refused: {error}\n")
    except FAILURES as error:
        sys.stderr.write(f"Refused: {type(error).__name__}: {error}\n")
    return 1


def _add(home: Home, options: argparse.Namespace) -> int:
    """Plug in an account, keeping only a reference to its credentials."""
    known = accounts.load(home.accounts)
    account = accounts.new_account(
        options.provider,
        name=options.name,
        credentials=_credentials(options),
        allowance=_decimal(options.allowance),
        paid_cap=_decimal(options.paid_cap),
        session_limit=options.session_limit,
        device=_device(options) if options.provider == "local" else None,
        expires=options.expires,
    )
    if any(item.name == account.name for item in known):
        raise ValueError(f"An account named {account.name} exists. Choose another --name.")
    accounts.save(home.accounts, [*known, account])
    sys.stdout.write(
        f"Added {account.name} ({account.provider}). Check it with: forge check {account.name}\n"
    )
    return 0


def _remove(home: Home, options: argparse.Namespace) -> int:
    """Unplug an account. Its past sessions stay in the usage log."""
    known = accounts.load(home.accounts)
    accounts.find(known, options.name)
    accounts.save(home.accounts, [item for item in known if item.name != options.name])
    sys.stdout.write(f"Removed {options.name}.\n")
    return 0


def _accounts(home: Home, options: argparse.Namespace) -> int:
    """Print each account, its device, the free time it has left and when that resets."""
    del options
    known = accounts.load(home.accounts)
    left = accounting.remaining(home, known, accounting.now())
    checks = json.loads(home.checks.read_text()) if home.checks.exists() else {}
    sys.stdout.write(
        f"{'Account':<12}{'Provider':<10}{'Free left':>14}  {'Resets (UTC)':<19}{'Check':<13}GPUs\n"
    )
    for account in known:
        sys.stdout.write(_account_row(account, left.get(account.name), checks.get(account.name)))
    return 0


def _account_row(
    account: accounts.Account, left: Remaining | None, check: dict[str, object] | None
) -> str:
    """One line of ``forge accounts``."""
    gpus = ", ".join(gpu.name for gpu in account.gpus)
    free, resets = "unlimited", "-"
    if left is not None and account.unit is not None:
        free = f"{max(left.amount - account.paid_cap, Decimal(0)):.2f} {_unit(account)}"
        if left.resets_at is not None:
            resets = left.resets_at.strftime("%Y-%m-%d %H:%M")
        elif left.expires is not None:
            resets = f"expires {left.expires.date().isoformat()}"
    result = "not checked" if check is None else ("ok" if check["ok"] else "failed")
    return f"{account.name:<12}{account.provider:<10}{free:>14}  {resets:<19}{result:<13}{gpus}\n"


def _gpus(home: Home, options: argparse.Namespace) -> int:
    """Print every GPU the plugged-in accounts offer: its memory, where, and its price."""
    del options
    offers: dict[str, list[tuple[accounts.Account, Gpu]]] = {}
    for account in accounts.load(home.accounts):
        for gpu in account.gpus:
            offers.setdefault(gpu.name, []).append((account, gpu))
    sys.stdout.write(f"{'GPU':<11}{'Memory':>7}  {'Accounts':<24}Price per hour\n")
    for name, offered in offers.items():
        memory = f"{offered[0][1].memory_gb} GB" if offered[0][1].memory_gb else "-"
        where = ", ".join(account.name for account, _ in offered)
        prices = ", ".join(_hourly(gpu) for _, gpu in offered)
        sys.stdout.write(f"{name:<11}{memory:>7}  {where:<24}{prices}\n")
    return 0


def _hourly(gpu: Gpu) -> str:
    """A GPU's price per hour in words: free, or dollars to the cent."""
    if gpu.usd_per_second == 0:
        return "free"
    return f"${gpu.usd_per_second * HOUR:.2f}"


def _check(home: Home, options: argparse.Namespace) -> int:
    """Try each account's credentials with a read-only call and remember the result."""
    known = accounts.load(home.accounts)
    chosen = [accounts.find(known, options.name)] if options.name else known
    results = json.loads(home.checks.read_text()) if home.checks.exists() else {}
    status = 0
    for account in chosen:
        try:
            note = for_account(account, home.state / "sessions").check(account)
            results[account.name] = {"ok": True, "note": note, "checked": runs.now_text()}
            sys.stdout.write(f"{account.name}: ok ({note})\n")
        except (ValueError, *FAILURES) as error:
            reason = str(error).splitlines()[0] if str(error) else type(error).__name__
            results[account.name] = {"ok": False, "note": reason, "checked": runs.now_text()}
            sys.stdout.write(f"{account.name}: failed ({reason})\n")
            status = 1
    home.checks.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    return status


def _run(home: Home, options: argparse.Namespace) -> int:
    """Start a run of a training script and follow it until it finishes or must wait."""
    job = jobs.describe(options.script, options.args, options.requirements)
    name = options.name or runs.name_for(options.script, accounting.now())
    request = Request(seconds=options.max_seconds, gpu=options.gpu, on=options.on)
    loop.start(home, job, request, name)
    sys.stdout.write(f"Run {name} started.\n")
    return loop.follow(home, name, sys.stdout, options.on)


def _status(home: Home, options: argparse.Namespace) -> int:
    """Print every run: its state, its last account, its sessions and the time they took."""
    del options
    sys.stdout.write(f"{'Run':<36}{'State':<10}{'Account':<12}{'Sessions':>8}{'Seconds':>9}\n")
    for row in control.status(home):
        sys.stdout.write(
            f"{row['run']:<36}{row['state']:<10}{row['account']:<12}"
            f"{row['sessions']:>8}{row['seconds']:>9}\n"
        )
    return 0


def _pause(home: Home, options: argparse.Namespace) -> int:
    """Pause a run at its next batch and keep its snapshot."""
    return control.pause(home, options.run, sys.stdout)


def _resume(home: Home, options: argparse.Namespace) -> int:
    """Continue a run, optionally on a chosen account."""
    return control.resume(home, options.run, sys.stdout, options.on)


def _logs(home: Home, options: argparse.Namespace) -> int:
    """Print a run's logs."""
    sys.stdout.write(control.logs(home, options.run))
    return 0


def _down(home: Home, options: argparse.Namespace) -> int:
    """Stop a run, or everything, and confirm nothing is left running."""
    return control.down(home, options.run, sys.stdout)


def _split(arguments: list[str]) -> tuple[list[str], list[str]]:
    """Split forge's own arguments from the job's, which follow the first ``--``."""
    if "--" not in arguments:
        return arguments, []
    index = arguments.index("--")
    return arguments[:index], arguments[index + 1 :]


def _credentials(options: argparse.Namespace) -> str | None:
    """The credential reference to store: a token file path or a Modal profile name."""
    if options.provider == "kaggle":
        return str(options.credentials or _default_kaggle_token())
    if options.provider == "modal":
        return options.profile
    return None


def _default_kaggle_token() -> Path:
    """Kaggle's own token file, ``~/.kaggle/access_token``, or else ``kaggle.json``."""
    folder = Path.home() / ".kaggle"
    token = folder / "access_token"
    return token if token.exists() else folder / "kaggle.json"


def _device(options: argparse.Namespace) -> str:
    """The device a local account uses: the one given, else the best one torch finds."""
    if options.device:
        return options.device
    import torch  # noqa: PLC0415 - torch is slow to import, so only when needed

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _date(value: str) -> date:
    """Parse a date option such as 2027-01-31."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date like 2027-01-31") from None


def _decimal(value: str | None) -> Decimal | None:
    """Parse an optional decimal option."""
    return Decimal(value) if value is not None else None


def _unit(account: accounts.Account) -> str:
    """A short label for the account's unit."""
    return "GPU-h" if account.unit is Unit.GPU_HOURS else "USD"


COMMANDS = {
    "add": _add,
    "remove": _remove,
    "accounts": _accounts,
    "gpus": _gpus,
    "check": _check,
    "run": _run,
    "status": _status,
    "pause": _pause,
    "resume": _resume,
    "logs": _logs,
    "down": _down,
}


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser for every command."""
    parser = argparse.ArgumentParser(prog="forge", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Plug in an account: kaggle, modal or local.")
    add.add_argument("provider", choices=accounts.PROVIDERS)
    add.add_argument("--name", help="A short name; defaults to the provider's name.")
    add.add_argument("--credentials", type=Path, help="Kaggle token file (default ~/.kaggle).")
    add.add_argument("--profile", help="Modal profile name (default: Modal's active profile).")
    add.add_argument("--allowance", help="Free amount per period, if not the provider default.")
    add.add_argument(
        "--expires",
        type=_date,
        metavar="DATE",
        help="Make the allowance a one-off credit, gone at 00:00 UTC on DATE (2027-01-31).",
    )
    add.add_argument("--paid-cap", help="Paid time forge may use per period (default 0).")
    add.add_argument("--session-limit", type=int, help="Longest session, in seconds.")
    add.add_argument("--device", choices=("cuda", "mps", "cpu"), help="For local accounts.")
    commands.add_parser("remove", help="Unplug an account.").add_argument("name")
    commands.add_parser("accounts", help="List accounts and the free time they have left.")
    commands.add_parser("gpus", help="List the GPUs your accounts offer, with their prices.")
    check = commands.add_parser("check", help="Check that accounts' credentials work.")
    check.add_argument("name", nargs="?")
    run = commands.add_parser(
        "run",
        help="Run a training script where there is time. Put its own arguments after --.",
    )
    run.add_argument("script", type=Path)
    run.add_argument("--on", help="Start on this account.")
    run.add_argument("--name", help="A name for the run.")
    run.add_argument("--max-seconds", type=int, help="Longest session this run may use.")
    run.add_argument(
        "--gpu",
        nargs="?",
        const=ANY,
        metavar="NAME",
        help="Run on this GPU (see forge gpus), or on any GPU when no name is given.",
    )
    run.add_argument("--requirements", type=Path, help="Extra pip requirements for clouds.")
    commands.add_parser("status", help="List runs.")
    for name, summary in (("pause", "Pause a run at a snapshot."), ("logs", "Print a run's logs.")):
        commands.add_parser(name, help=summary).add_argument("run")
    resume = commands.add_parser("resume", help="Continue a run.")
    resume.add_argument("run")
    resume.add_argument("--on", help="Continue on this account.")
    down = commands.add_parser("down", help="Stop a run, or all, and confirm shutdown.")
    down.add_argument("run", nargs="?")
    return parser


if __name__ == "__main__":
    sys.exit(main())
