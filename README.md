# forge

Plug in your GPU accounts, run your training script, and forge keeps it going.

forge runs a training job across whatever compute you have: a Kaggle account, a Modal
account, your own machine. It starts the job where there is free GPU time. When a session
ends or the free time runs out, it saves a snapshot and carries on somewhere else from the
exact step it stopped at. You can pause a run and resume it on another account whenever
you like, and the script itself stays an ordinary PyTorch script.

It is for students, researchers and anyone whose compute comes in pieces: a free weekly
quota here, some credit there, a laptop. Out of the box it only uses free time. Built in
South Africa.

```text
forge add kaggle                  # plug in an account
forge add modal
forge add local                   # this machine
forge accounts                    # what is plugged in and how much free time is left
forge run train.py -- --epochs 3  # runs where there is time; moves when a session ends
forge status                      # every run, its state and the account it is on
forge pause RUN / forge resume RUN [--on ACCOUNT]
forge logs RUN
forge down [RUN]                  # stop, and confirm nothing is left running
```

## Quickstart

You need [uv](https://docs.astral.sh/uv/) on macOS or Linux. This needs no account, and
no network after `uv sync`. The install is mostly torch: a minute on a Mac, about five
minutes and 5 GB on Linux, where pip brings the CUDA build. The demo itself takes under
a minute:

```bash
uv sync
scripts/demo-hop.sh
```

The demo plugs in two "local" accounts that each allow sessions of only 10 seconds, then
runs `examples/tiny/train.py`, an ordinary PyTorch script. The job has to hop between the
two accounts to finish, and its final weights must match an uninterrupted run. How many
sessions it takes depends on how quickly the machine starts Python; this is one run:

```text
Account     Provider  GPU        Free left  Resets (UTC)     Check
first       local     cpu        unlimited  -                not checked
second      local     cpu        unlimited  -                not checked
Run hop started.
Session 1 on first (up to 10 s).
Snapshot saved after step 6 in 0.016s.
Session 2 on second (up to 10 s).
Snapshot saved after step 13 in 0.013s.
Session 3 on first (up to 10 s).
Snapshot saved after step 19 in 0.005s.
Session 4 on second (up to 10 s).
ad73537a4f441f869c3a176388ed099b1c7395269cbab297e216f2511944f70b
Run hop finished after 4 sessions. Outputs: .../state/runs/hop/outputs
Run                                 State     Account     Sessions  Seconds
hop                                 finished  second             4       32
hopped        ad73537a4f441f869c3a176388ed099b1c7395269cbab297e216f2511944f70b
uninterrupted ad73537a4f441f869c3a176388ed099b1c7395269cbab297e216f2511944f70b
Same weights after the hops.
```

The demo keeps everything in a temporary `FORGE_HOME`. Without it, forge keeps accounts
in `~/.config/forge/accounts.toml` and its state in `~/.local/share/forge/`.

## Across two clouds

The same job on real accounts, using free time only. It starts on a Modal T4 with
sessions capped at 150 seconds, saves at step 11 and finishes in a Kaggle notebook:

```text
$ forge run examples/tiny/train.py --on modal --name core-test-live -- --step-delay 10
Run core-test-live started.
Session 1 on modal (up to 150 s).
Snapshot saved after step 11 in 0.009s.
Session 2 on kaggle (up to 900 s).
Run core-test-live finished after 2 sessions. Outputs: .../state/runs/core-test-live/outputs
$ forge logs core-test-live
--- session 1 on modal
Snapshot saved after step 11 in 0.009s.
--- session 2 on kaggle
7be3a0ccd35bf012607ca83a124d1d2e97fa56722475495c3c35f53a9bf36bae
```

The script run straight through on Modal, with no stop, prints the same hash. The job
changed provider halfway and its final weights did not change by a single bit.

## Plug in an account

**This machine.** `forge add local`. forge uses CUDA or the Apple GPU if torch finds one,
else the CPU; choose with `--device cpu|cuda|mps`. `--session-limit SECONDS` makes forge
move the job on after that long.

**Kaggle.** Create an API token on your Kaggle account settings page and save it as
`~/.kaggle/access_token` (or `~/.kaggle/kaggle.json`), readable by you only
(`chmod 600`). Then `forge add kaggle` and `forge check kaggle`. Another location:
`--credentials PATH`.

**Modal.** Run `modal token new` once, which writes your Modal profile. Then
`forge add modal` and `forge check modal`. Another profile: `--profile NAME`. Set your
workspace's spend limit in Modal as well.

forge stores where your credentials are, never the secrets. Each account starts with its
provider's free allowance (Kaggle: 30 GPU-hours a week; Modal: $30 a month) and a paid
cap of 0. `--allowance` and `--paid-cap` change them.

## How a job moves

1. forge packs the script's folder (leaving out caches, environments and `outputs/`).
2. It picks an account: a local GPU that fits first, then free time that resets
   soonest, then paid time, only where you set a paid cap. The session gets the
   shortest of your `--max-seconds`, the account's session limit and the time left.
3. It holds that time against the account, then starts the session with a limit the
   provider enforces itself.
4. Inside the session, a wrapper runs your script unchanged. Shortly before the limit
   it lets the current batch finish and saves a snapshot: weights, optimiser, random
   streams and the exact position in the shuffled data.
5. forge fetches the snapshot, confirms the session stopped and writes the time it took
   to the usage log, which ends the hold.
6. The next session, on the next account, restores the snapshot and carries on. When
   the script finishes, its `outputs/` come back to you.

If no account has time left, the run waits and forge says when the next allowance
resets; `forge resume RUN` continues it. Ctrl-C detaches without stopping the session.
See [docs/snapshots.md](docs/snapshots.md) for what a script must look like.

## Safety

- Free time only, unless you give an account a paid cap.
- Every session has a time limit enforced by its provider, even if forge is closed.
- A session counts as stopped only once its provider confirms it.
- Every session goes into a usage log in forge's state folder.
- Cloud calls run in a short-lived process that holds one account's credentials.

## What works today

Covered by the offline test suite and the demo above:

- Accounts, placement, holds and the usage log.
- `local` sessions: run, hop at session limits, pause, resume, wait when no account
  has time, and `down`.
- Snapshots that resume bit for bit on the same machine, also across data worker counts
  and between CPU and the Apple GPU (see [docs/snapshots.md](docs/snapshots.md)).

Run against live accounts, on free time only:

- A job that starts on a Modal T4, saves at its session limit and finishes on Kaggle,
  with the same final weights as the uninterrupted run (the transcript above).
- `forge pause` from a second terminal while `forge run` is attached to a Modal session,
  then `forge resume`.

The Kaggle and Modal adapters are young: two providers, one GPU type each, and their
tests otherwise run against fakes of the SDKs. Kaggle notebooks cannot be paused from
forge; they save at their own limit. See [docs/providers.md](docs/providers.md).

Not built yet: other providers, running several sessions of one job at once, and
measured billing (a session counts its wall-clock time).

## Layout

```text
src/forge/
  cli.py, loop.py, control.py   the commands and the run loop
  accounts.py, home.py          accounts and where forge keeps its files
  jobs.py, runs.py              jobs, and the journal of each run
  accounting.py                 time left per account, holds and the usage log
  core/                         pure maths: time left, placement
  ledger.py, files.py           the usage log, and files written so a crash cannot break them
  providers/                    local, kaggle and modal, behind one interface
  session.py, runner.py         what runs inside a session, around your script
  snapshot/                     resumable training state
```

See [docs/architecture.md](docs/architecture.md) for how the pieces fit.

## Checks

```bash
scripts/check.sh
```

This runs ruff, pyright, import-linter, a small repo lint and the test suite, offline.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
