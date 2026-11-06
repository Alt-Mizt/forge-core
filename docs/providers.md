# Providers

A provider is somewhere a session can run. Each one implements the same small
interface (`forge/providers/__init__.py`): check an account, start a session with a
hard time limit, poll it, fetch its results, show its log, ask it to pause, and stop it
with confirmed shutdown.

Each provider also has a GPU catalogue in `forge/accounts.py`: every GPU a session can
ask for, with its memory and the most it costs per second, cheapest first. `forge gpus`
shows the catalogues of your accounts. `forge run --gpu NAME` picks a GPU by its
catalogue name, in any case; without a name, a session gets the cheapest GPU that fits.

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

- GPU catalogue: one entry named after the device, `cuda`, `mps` or `cpu`, with no price.
  `cpu` does not count as a GPU for `--gpu`. forge does not measure the device's memory,
  so a job that declares `memory_gb` in its `forge.toml` passes over local accounts.

## kaggle

A session is a private notebook on Kaggle's T4 machine, with Kaggle's own session
timeout set to the session length. The notebook carries the session folder inside its
code, installs the needed torch with pip, and runs the session. forge reads the job's
exit status from the notebook's output once it ends.

- GPU catalogue: `T4`, 16 GB, free. A session gets the machine with two T4s; Kaggle's
  free tier has no other GPU forge supports.
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

A session is a sandbox with one GPU, at most 4 CPU cores and 32 GiB of memory, no
network, the gVisor runtime and a timeout that Modal enforces. The sandbox image has the
needed torch installed. After the job ends the sandbox waits up to two minutes for forge
to collect the results; that time is held and counted too.

- GPU catalogue, from Modal's pricing page on 6 November 2026. Modal bills a sandbox for
  its GPU plus the larger of the CPU and memory it asked for or used, at three times
  list price ($0.00003942 a core-second, $0.00000667 a GiB-second). forge counts every
  second at the GPU's price plus the sandbox's whole CPU and memory cap, $0.00037112:

  | GPU | Memory | Modal's GPU price per second | forge counts per second |
  |---|---|---|---|
  | `T4` | 16 GB | $0.000164 | $0.00053512 |
  | `L4` | 24 GB | $0.000222 | $0.00059312 |
  | `A10` | 24 GB | $0.000306 | $0.00067712 |
  | `L40S` | 48 GB | $0.000542 | $0.00091312 |
  | `A100-40GB` | 40 GB | $0.000583 | $0.00095412 |
  | `A100-80GB` | 80 GB | $0.000694 | $0.00106512 |
  | `H100` | 80 GB | $0.001097 | $0.00146812 |
  | `H200` | 141 GB | $0.001261 | $0.00163212 |

  Modal also offers the RTX PRO 6000, B200 and B300; they are left out because Modal's
  pages do not state their memory. Modal may run an `H100` request on an H200, at the
  H100's price.

- Credentials: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or a Modal profile name
  (`forge add modal --profile NAME`), which the Modal SDK reads from its own config.
- Allowance: $30 of credit a month, spent at the per-second amounts above.
- `forge pause` sends SIGTERM to the job inside the sandbox.
- Set the workspace's spend limit in Modal as well. forge's paid cap only limits what
  forge itself starts.

## What every cloud call does

Each SDK call runs in a short-lived worker process (`providers/bridge.py` starts
`providers/worker.py`). The worker gets a small fixed environment plus one account's
credentials, discards the SDK's own logs and error text, and returns selected fields as
JSON. A Kaggle worker also gets an empty config folder, so it cannot fall back to
another Kaggle login.
