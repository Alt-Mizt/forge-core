# Providers

A provider is somewhere a session can run. Each one implements the same small
interface (`forge/providers/__init__.py`): check an account, start a session with a
hard time limit, poll it, fetch its results, show its log, ask it to pause, and stop it
with confirmed shutdown.

Every session runs `forge/session.py` in a folder holding the job, forge's wrapper and,
when resuming, the snapshot. It writes `exit_code`, `log.txt`, `outputs.zip` and, when
the job paused, `snapshot/`. That folder travels as one zip file, so all three
providers run exactly the same code.

## local

Runs the session as a detached process on this machine, in a folder under forge's state.
The session runner enforces the time limit itself, so a session stays bounded even when
forge is not watching. `forge pause` sends SIGTERM to the job; `forge down` kills it and
waits until the processes are gone. Set `--device cpu|cuda|mps` when adding the account;
forge passes it to the job as `FORGE_DEVICE`. A local account has no allowance unless
you give it one.

## kaggle

A session is a private notebook on Kaggle's T4 machine, with Kaggle's own session
timeout set to the session length. The notebook carries the session folder inside its
code, installs the needed torch with pip, and runs the session. forge reads the job's
exit status from the notebook's output once it ends.

- Credentials: a token file, `~/.kaggle/access_token` or `~/.kaggle/kaggle.json` by
  default. It must be readable by you only.
- Allowance: 30 GPU-hours a week, reset on Saturday. A session's time counts once,
  although the machine has two T4s.
- One session at a time. A running notebook cannot be paused or cancelled from forge;
  it saves and ends at its own limit, and `forge down` waits for that.
- A notebook's log is available only after it ends.
- forge refuses to start a notebook if the free quota cannot cover its whole time
  limit, or if paid scaling is switched on.

## modal

A session is a sandbox with one T4, no network, the gVisor runtime and a timeout that
Modal enforces. The sandbox image has the needed torch installed. After the job ends the
sandbox waits up to two minutes for forge to collect the results; that time is held and
counted too.

- Credentials: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or a Modal profile name
  (`forge add modal --profile NAME`), which the Modal SDK reads from its own config.
- Allowance: $30 of credit a month. forge bounds a T4 sandbox at $0.001 a second.
- `forge pause` sends SIGTERM to the job inside the sandbox.
- Set the workspace's spend limit in Modal as well. forge's paid cap only limits what
  forge itself starts.

## What every cloud call does

Each SDK call runs in a short-lived worker process (`providers/bridge.py` starts
`providers/worker.py`). The worker gets a small fixed environment plus one account's
credentials, discards the SDK's own logs and error text, and returns selected fields as
JSON. A Kaggle worker also gets an empty config folder, so it cannot fall back to
another Kaggle login.
