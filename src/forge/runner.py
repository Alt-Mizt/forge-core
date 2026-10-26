"""Run an unmodified PyTorch training script so that it can stop and resume at any step.

The wrapper watches the script build its model, optimiser, scheduler and DataLoader. On
SIGTERM, or when the session's deadline passes, it saves a snapshot after the current batch
and exits with status 143. Started with a snapshot, the same script resumes from it.

Usage: python -m forge.runner --snapshot DIR [--resume DIR] -- python train.py [arguments]
The deadline, a Unix time, comes from the ``FORGE_DEADLINE`` environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import signal
import sys
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, cast

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from forge.snapshot import Resumable, load, save

__all__ = ["run"]

TERMINATED = 143
FAILED = 1
SNAPSHOT_SECONDS = 60
MIN_COMMAND_PARTS = 2
PYTHON_NAMES = frozenset({"python", "python3", "python3.11"})


def main() -> int:
    """Run the wrapper from the command line and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    deadline = os.environ.get("FORGE_DEADLINE")
    try:
        return run(command, options.snapshot, options.resume, deadline=_deadline(deadline))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        sys.stderr.write(f"Refused: {error}\n")
    return FAILED


def run(
    command: list[str], snapshot: Path, resume: Path | None, *, deadline: float | None = None
) -> int:
    """Run a Python training script in this process, and save it to ``snapshot`` on SIGTERM.

    On SIGTERM, or at the first batch boundary after ``deadline`` (a Unix time), the current
    batch finishes, the snapshot is saved and the process exits with status 143. Every
    process trains at least one batch first, so a run always makes progress. With
    ``resume``, training continues from that snapshot instead.

    Supported loops have one model, one optimiser, at most one scheduler and one shuffled
    DataLoader. Each batch must finish within 60 seconds. Anything else refuses before the
    first batch.
    """
    script = _script(command)
    _prepare_parent(snapshot)
    if snapshot.exists():
        raise ValueError("The snapshot destination already exists; choose a new --snapshot.")
    capture = _Capture(snapshot, resume, deadline)
    original_arguments = sys.argv
    original_path = sys.path.copy()
    previous_handler = signal.signal(signal.SIGTERM, capture.request_stop)
    try:
        sys.argv = [str(script), *command[2:]]
        sys.path.insert(0, str(script.parent))
        try:
            with capture.instrument():
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code in (None, 0) and capture.requested_at is not None:
                capture.finish()
            raise
        if capture.requested_at is not None:
            capture.finish()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        sys.argv = original_arguments
        sys.path[:] = original_path
    return 0


def _deadline(value: str | None) -> float | None:
    """Parse the ``FORGE_DEADLINE`` value, if there is one."""
    return float(value) if value else None


def _script(command: list[str]) -> Path:
    """Return the training script named by ``python path/to/training.py [arguments]``."""
    if len(command) < MIN_COMMAND_PARTS or Path(command[0]).name not in PYTHON_NAMES:
        raise ValueError("Give the script as: -- python path/to/training.py [arguments].")
    script = Path(command[1]).resolve()
    if not script.is_file():
        raise ValueError("The training script does not exist.")
    return script


class _Capture:
    """Find the script's training objects as it creates them, and save them on request."""

    def __init__(self, snapshot: Path, resume: Path | None, deadline: float | None) -> None:
        """Start with nothing found; objects are recorded as the script creates them."""
        self.snapshot = snapshot
        self.resume = resume
        self.deadline = deadline
        self.trained = 0
        self.modules: list[weakref.ReferenceType[torch.nn.Module]] = []
        self.optimizers: list[weakref.ReferenceType[Optimizer]] = []
        self.schedulers: list[weakref.ReferenceType[LRScheduler]] = []
        self.loader: DataLoader[Any] | None = None
        self.state: Resumable | None = None
        self.epoch_calls = 0
        self.requested_at: float | None = None
        self.original_iter = DataLoader.__iter__

    def request_stop(self, signum: int, frame: FrameType | None) -> None:
        """Note the SIGTERM. Saving waits for the next batch boundary, never mid-update."""
        del signum, frame
        if self.requested_at is None:
            self.requested_at = time.monotonic()

    def instrument(self) -> ExitStack:
        """Record new models, optimisers and schedulers, and route DataLoader iteration here."""
        stack = ExitStack()
        original_module = cast(Callable[..., None], torch.nn.Module.__init__)
        original_optimizer = cast(Callable[..., None], Optimizer.__init__)
        original_scheduler = cast(Callable[..., None], LRScheduler.__init__)

        def module_init(instance: torch.nn.Module, *args: object, **kwargs: object) -> None:
            original_module(instance, *args, **kwargs)
            self.modules.append(weakref.ref(instance))

        def optimizer_init(instance: Optimizer, *args: object, **kwargs: object) -> None:
            original_optimizer(instance, *args, **kwargs)
            self.optimizers.append(weakref.ref(instance))

        def scheduler_init(instance: LRScheduler, *args: object, **kwargs: object) -> None:
            original_scheduler(instance, *args, **kwargs)
            self.schedulers.append(weakref.ref(instance))

        def iterate(loader: DataLoader[Any]) -> Iterator[Any]:
            return self.iterate(loader)

        stack.enter_context(_replaced(torch.nn.Module, "__init__", module_init))
        stack.enter_context(_replaced(Optimizer, "__init__", optimizer_init))
        stack.enter_context(_replaced(LRScheduler, "__init__", scheduler_init))
        stack.enter_context(_replaced(DataLoader, "__iter__", iterate))
        return stack

    def iterate(self, loader: DataLoader[Any]) -> Iterator[Any]:
        """Iterate the script's DataLoader, skipping the epochs a resume already finished."""
        if self.state is None:
            self.bind(loader)
        if loader is not self.loader or self.state is None:
            raise ValueError("The wrapper supports one training DataLoader only.")
        requested_epoch = self.epoch_calls
        self.epoch_calls += 1
        if requested_epoch < self.state.epoch:
            return iter(())
        return self.batches()

    def bind(self, loader: DataLoader[Any]) -> None:
        """Bind the one optimiser, the one model owning its parameters, and any scheduler."""
        optimizers = [item for reference in self.optimizers if (item := reference()) is not None]
        if len(optimizers) != 1:
            raise ValueError("The wrapper requires exactly one optimizer.")
        optimizer = optimizers[0]
        parameters = {id(value) for group in optimizer.param_groups for value in group["params"]}
        models = [
            model
            for reference in self.modules
            if (model := reference()) is not None
            and {id(value) for value in model.parameters()} == parameters
        ]
        roots = [
            model
            for model in models
            if not any(model in tuple(other.children()) for other in models if other is not model)
        ]
        schedulers = [item for reference in self.schedulers if (item := reference()) is not None]
        if len(roots) != 1 or len(schedulers) > 1:
            raise ValueError("The wrapper requires one complete model and at most one scheduler.")
        self.loader = loader
        self.state = Resumable(roots[0], optimizer, loader, schedulers[0] if schedulers else None)
        if self.resume is not None:
            document = load(self.resume)
            if document.get("script_sha256") != _script_digest():
                raise ValueError("The training script changed since this snapshot.")
            self.state.restore_state(document)

    def batches(self) -> Iterator[Any]:
        """Yield batches, saving and exiting first if SIGTERM arrived during the last one."""
        if self.state is None:
            raise ValueError("No supported training loop was discovered.")
        iterator = iter(self.state)
        while True:
            if self._past_deadline():
                self.request_stop(signal.SIGTERM, None)
            if self.requested_at is not None:
                self.finish()
            # Resumable creates its internal loader lazily. Bypass interception
            # during its fetch, so only the script's loader is counted as an epoch.
            try:
                with _replaced(DataLoader, "__iter__", self.original_iter):
                    batch = next(iterator)
            except StopIteration:
                return
            yield batch
            self.trained += 1

    def _past_deadline(self) -> bool:
        """Whether the session deadline has passed, once this process has trained a batch."""
        return self.deadline is not None and self.trained > 0 and time.time() >= self.deadline

    def finish(self) -> None:
        """Save the snapshot and exit with status 143; refuse if saving took past the deadline."""
        if self.state is None or self.requested_at is None:
            raise ValueError("SIGTERM arrived before a supported training batch boundary.")
        document = self.state.state_dict()
        document["script_sha256"] = _script_digest()
        save(document, self.snapshot)
        elapsed = time.monotonic() - self.requested_at
        if elapsed >= SNAPSHOT_SECONDS:
            raise ValueError("The batch exceeded the 60-second snapshot deadline.")
        sys.stderr.write(f"Snapshot saved after step {self.state.step} in {elapsed:.3f}s.\n")
        raise SystemExit(TERMINATED)


@contextmanager
def _replaced(owner: type, name: str, value: object) -> Iterator[None]:
    """Swap one attribute of a class for the duration, then put the original back."""
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _prepare_parent(snapshot: Path) -> None:
    """Refuse a snapshot destination whose parent folder does not exist."""
    if not snapshot.parent.is_dir():
        raise ValueError("Create the --snapshot parent folder before starting the run.")


def _script_digest() -> str:
    """Hash the wrapped script's bytes, so a resume can refuse a changed script."""
    return hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
