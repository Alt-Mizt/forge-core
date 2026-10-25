"""Snapshots of a training loop's full state that resume in a fresh process, on other hardware.

A snapshot holds the model, optimiser, scheduler, every random stream and the exact position in
the shuffled data, written as JSON plus safetensors. Nothing is ever unpickled.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, cast

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, RandomSampler

from forge.snapshot import random_state
from forge.snapshot.codec import load, save
from forge.snapshot.data_order import (
    NEEDS_STANDARD_LOADER,
    NEEDS_STANDARD_SAMPLER,
    SAME_AUGMENTATION_SEED,
    SAME_SAMPLER_SEED,
    WorkerPool,
    epoch_order,
    indexed,
    validate_workers,
)

LOOP_VERSION = 2

__all__ = ["Resumable", "load", "resumable", "save"]


class Resumable:
    """Track completed batches for ordinary or explicitly indexed data.

    Iterate this object instead of the original loader, and save after an optimizer
    and scheduler step. Workers require the stateless epoch/index contract; custom
    samplers are refused. The dataset must be unchanged on resume.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        data: DataLoader[Any],
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Bind the objects whose state must advance together."""
        if type(data) is not DataLoader:
            raise ValueError(NEEDS_STANDARD_LOADER)
        self.indexed_data = indexed(data.dataset)
        validate_workers(data, self.indexed_data)
        if type(data.sampler) is not RandomSampler:
            raise ValueError(NEEDS_STANDARD_SAMPLER)
        if data.generator is None or data.batch_size is None or data.sampler.replacement:
            raise ValueError("Resume requires an explicit generator and fixed batch size.")
        if data.sampler.num_samples != len(cast(Sized, data.dataset)):
            raise ValueError("Resume requires a complete permutation of the dataset.")
        if data.sampler.generator is not data.generator:
            raise ValueError("The sampler and DataLoader must share their generator.")
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.scheduler = scheduler
        self.epoch = 0
        self.step = 0
        self.position = 0
        self.order: list[int] | None = None
        self.sampler_exhausted = False
        self.order_seed = data.generator.initial_seed()
        self.worker_pool: WorkerPool | None = None

    def __iter__(self) -> Iterator[Any]:
        """Yield the epoch's remaining batches, counting each one as a completed step."""
        generator = self.data.generator
        if generator is None or self.data.batch_size is None:
            raise ValueError("The bound DataLoader configuration changed.")
        batch_size = self.data.batch_size
        order = self._epoch_order(generator)
        batches = self._remaining_batches(order, batch_size)
        iterator = self._fetch(generator, batches)
        for indices in batches:
            short = len(indices) < batch_size
            if self.indexed_data is None and short and not self.sampler_exhausted:
                # The plain sampler draws its end-of-data permutation before a short batch.
                torch.randperm(len(order), generator=generator)
                self.sampler_exhausted = True
            batch = next(iterator)
            self.position += len(indices)
            self.step += 1
            yield batch
        self._finish_epoch(generator, len(order))

    def _epoch_order(self, generator: torch.Generator) -> list[int]:
        """Return this epoch's sample order, drawing it the way the original loader would."""
        if self.order is None and self.indexed_data is not None:
            self.order = epoch_order(self.order_seed, self.epoch, self._size())
        if self.order is None:
            # Match the original loader's worker-seed draw before its first permutation.
            iter(self.data)
            self.order = torch.randperm(self._size(), generator=generator).tolist()
        return self.order

    def _remaining_batches(self, order: list[int], batch_size: int) -> list[list[int]]:
        """Split the not-yet-trained part of the order into batches, honouring drop_last."""
        return [
            order[start : start + batch_size]
            for start in range(self.position, len(order), batch_size)
            if not self.data.drop_last or start + batch_size <= len(order)
        ]

    def _fetch(self, generator: torch.Generator, batches: list[list[int]]) -> Iterator[Any]:
        """Return an iterator that loads exactly these batches, without moving the shuffle."""
        # Creating a loader iterator draws a worker seed even with no workers. Restore the
        # generator afterwards so a resumed iterator does not advance the shuffle stream.
        generator_state = generator.get_state()
        if self.indexed_data is not None:
            if self.worker_pool is None:
                self.worker_pool = WorkerPool(self.data)
            iterator = self.worker_pool.fetch(self.epoch, batches) if batches else iter(())
        else:
            loader = DataLoader(
                self.data.dataset,
                batch_sampler=batches,
                collate_fn=self.data.collate_fn,
                generator=generator,
            )
            iterator = iter(loader)
        generator.set_state(generator_state)
        return iterator

    def _finish_epoch(self, generator: torch.Generator, size: int) -> None:
        """Advance to the next epoch, keeping the random stream in step with the plain loader."""
        # RandomSampler draws one final (empty-slice) permutation on exhaustion.
        # Keeping that draw preserves the unwrapped loop's next epoch order.
        if self.indexed_data is None and not self.sampler_exhausted:
            torch.randperm(size, generator=generator)
        self.sampler_exhausted = False
        self.epoch += 1
        self.position = 0
        self.order = None

    def _size(self) -> int:
        """The number of samples in the bound dataset."""
        return len(cast(Sized, self.data.dataset))

    def save(self, folder: str | Path) -> None:
        """Save at a completed batch boundary to a new snapshot folder."""
        save(self.state_dict(), folder)

    def state_dict(self) -> dict[str, Any]:
        """Return all completed loop state for atomic publication with provenance."""
        generator = self.data.generator
        if generator is None:
            raise ValueError("The DataLoader generator is missing.")
        return {
            "loop_version": LOOP_VERSION,
            "index_seed": self.indexed_data.seed if self.indexed_data else None,
            "order_seed": self.order_seed,
            "sampler_exhausted": self.sampler_exhausted,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "rng": random_state.capture(device=next(self.model.parameters()).device.type),
            "data_generator": generator.get_state(),
            "epoch": self.epoch,
            "step": self.step,
            "position": self.position,
            "order": self.order,
            "data_size": self._size(),
            "batch_size": self.data.batch_size,
            "drop_last": self.data.drop_last,
            "torch_version": str(torch.__version__),
        }

    def restore(self, folder: str | Path) -> None:
        """Load the same loop, moving optimizer moments to its model's target device."""
        self.restore_state(load(folder))

    def restore_state(self, state: dict[str, Any]) -> None:
        """Apply an already loaded and verified snapshot to the bound objects."""
        self._require_compatible(state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.epoch, self.step = state["epoch"], state["step"]
        self.position, self.order = state["position"], state["order"]
        self.sampler_exhausted = state["sampler_exhausted"]
        generator = self.data.generator
        if generator is None:
            raise ValueError("The DataLoader generator is missing.")
        generator.set_state(state["data_generator"])
        random_state.restore(state["rng"])

    def _require_compatible(self, state: dict[str, Any]) -> None:
        """Refuse a snapshot taken with a different loop, data setup, scheduler or torch."""
        if state.get("loop_version") != LOOP_VERSION:
            raise ValueError("Snapshot uses older loader semantics; keep its original loop build.")
        seed = self.indexed_data.seed if self.indexed_data else None
        if state.get("index_seed") != seed:
            raise ValueError(SAME_AUGMENTATION_SEED)
        if self.indexed_data is not None and state.get("order_seed") != self.order_seed:
            raise ValueError(SAME_SAMPLER_SEED)
        if _plain(state["torch_version"]) != _plain(str(torch.__version__)):
            raise ValueError(
                f"This snapshot was written with torch {state['torch_version']}; install that "
                "version to resume it."
            )
        configuration = (self._size(), self.data.batch_size, self.data.drop_last)
        if configuration != (state["data_size"], state["batch_size"], state["drop_last"]):
            raise ValueError("The snapshot's data configuration differs from this loader.")
        if (self.scheduler is None) != (state["scheduler"] is None):
            raise ValueError("Restore the same scheduler configuration as the snapshot.")


def _plain(version: str) -> str:
    """A torch version without its build tag, so CPU and CUDA builds of one release match."""
    return version.split("+", 1)[0]


def resumable(
    model: torch.nn.Module,
    optimizer: Optimizer,
    data: DataLoader[Any],
    scheduler: LRScheduler | None = None,
) -> Resumable:
    """Bind a handwritten loop to snapshot model, optimizer, scheduler and data."""
    return Resumable(model, optimizer, data, scheduler)
