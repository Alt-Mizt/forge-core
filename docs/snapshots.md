# Snapshots

A job can stop at any completed training step and continue in a fresh process, on
this machine or another one. The snapshot carries everything the next step needs.

## What a snapshot holds

A snapshot is a folder with two files:

- `tensors.safetensors`: model weights, optimiser moments and the torch random
  streams (CPU, and CUDA or MPS when in use).
- `snapshot.json`: everything else: epoch, step, the exact shuffled sample order and
  the position in it, the scheduler, Python and NumPy random state, the torch version,
  and the SHA-256 of the script that wrote it.

Nothing is pickled, so loading a snapshot cannot run code. The JSON records a SHA-256
of its own contents and of the tensor file; loading checks both. Saving writes to a
hidden folder and publishes it with one rename, so a half-written snapshot never
appears.

## The contract with your script

forge runs your script, unchanged, inside a wrapper (`forge/runner.py`). The wrapper
imports the script as `__main__` in its own process and, while it runs, stands in for
the constructors of `torch.nn.Module`, `Optimizer` and `LRScheduler` and for
`DataLoader.__iter__`. That is how it sees the model, optimiser, scheduler and loader the
script creates, and how it sits inside the training loop at every batch, without the
script importing anything. Then:

1. On SIGTERM, or when the session's deadline passes, it lets the current batch
   finish, saves a snapshot and exits with status 143.
2. Started with a snapshot, it restores every object and the data position before the
   first batch, so the script carries on where it stopped.
3. Exit status 0 means the script finished. Anything it wrote under `outputs/` in its
   folder comes back to you.
4. Each session trains at least one batch before it honours the deadline, so a run
   always makes progress.

You can run the wrapper by hand:

```bash
uv run python -m forge.runner --snapshot /tmp/paused -- python examples/tiny/train.py --step-delay 0.2
# send it SIGTERM; it prints "Snapshot saved after step N" and exits 143
uv run python -m forge.runner --snapshot /tmp/again --resume /tmp/paused -- python examples/tiny/train.py
```

The wrapper supports one model holding all the optimiser's parameters, one optimiser,
at most one scheduler, and one shuffled map-style DataLoader with an explicit generator
and a fixed batch size. Anything else is refused with a message rather than guessed at.
A script that builds its loop differently can use the library directly:
`forge.resumable(model, optimizer, data, scheduler)` (see `examples/train_tiny.py`).

## Torch versions

A snapshot resumes only under the torch release that wrote it. forge picks the version
for each session in this order: the snapshot's, then a `torch==` pin in the job's
requirements, then the version installed here. Cloud sessions install that version;
a local session refuses if this machine has a different one, and says which to install.
A pin that contradicts the snapshot refuses too, and names the version to pin.

Results are bit for bit only on the same hardware and torch build, with deterministic
algorithms and one torch thread. Across CPU and GPU they are close; the demos below
print the difference.

## Demos

- `scripts/demo-resume.sh` stops `examples/train_tiny.py` after steps 1, 6 and 13 and
  checks that each fresh-process resume ends with the same weights.
- `scripts/demo-wrapper.sh` does the same through the wrapper, with SIGTERM.
- `scripts/demo-workers.sh` stops a four-worker data loader and resumes with two.
- `scripts/demo-cpu-to-gpu.sh` moves a run from CPU to the Apple GPU and back (macOS).

Datasets that declare `resumable_by_index = True` and an integer `seed` may use up to
four data workers; they receive `(epoch, index)` keys and must derive every random
augmentation from `(seed, epoch, index)`.
