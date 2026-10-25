"""Keep indexed augmentation and batch order independent of the data worker count."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, Sampler, default_collate

MAX_WORKERS = 4
SEED_RANGE = 2**64

# Why a DataLoader cannot be resumed, said once and reused.
NEEDS_INDEXED_DATASET = (
    "Data workers need a dataset with resumable_by_index=True and augmentation keyed by "
    "(seed, epoch, index)."
)
NEEDS_TORCH_SEED = "Indexed augmentation needs an integer seed in the torch range."
TOO_MANY_WORKERS = f"Use at most {MAX_WORKERS} data workers."
NEEDS_DEFAULT_COLLATION = "Indexed resume needs default collation and no worker init hook."
NEEDS_ORDERED_DELIVERY = "Resume needs batches delivered in sampler order."
NEEDS_STANDARD_LOADER = "Resume needs a standard map-style DataLoader."
NEEDS_STANDARD_SAMPLER = "Resume needs the standard shuffled sampler, not a custom sampler."
SAME_AUGMENTATION_SEED = "Keep the same indexed augmentation seed and dataset when resuming."
SAME_SAMPLER_SEED = "Keep the same DataLoader generator seed when resuming."


@runtime_checkable
class IndexedData(Protocol):
    """A dataset accepting (epoch, index), with randomness keyed only by seed and that key."""

    seed: int
    resumable_by_index: bool


def indexed(dataset: object) -> IndexedData | None:
    """Recognise the explicit stateless-index contract; ordinary datasets remain unchanged."""
    if isinstance(dataset, IndexedData) and dataset.resumable_by_index is True:
        if type(dataset.seed) is not int or not 0 <= dataset.seed < SEED_RANGE:
            raise ValueError(NEEDS_TORCH_SEED)
        return dataset
    return None


def validate_workers(data: DataLoader[Any], dataset: IndexedData | None) -> None:
    """Refuse worker-local randomness and unordered delivery rather than guessing state."""
    if not 0 <= data.num_workers <= MAX_WORKERS:
        raise ValueError(TOO_MANY_WORKERS)
    if data.num_workers and dataset is None:
        raise ValueError(NEEDS_INDEXED_DATASET)
    if dataset is not None and (data.collate_fn is not default_collate or data.worker_init_fn):
        raise ValueError(NEEDS_DEFAULT_COLLATION)
    if not getattr(data, "in_order", True):
        raise ValueError(NEEDS_ORDERED_DELIVERY)


def epoch_order(seed: int, epoch: int, size: int) -> list[int]:
    """Compute one seeded permutation in the main process, without consuming worker RNG."""
    generator = torch.Generator().manual_seed((seed + epoch) % SEED_RANGE)
    return torch.randperm(size, generator=generator).tolist()


class EpochBatches(Sampler[list[tuple[int, int]]]):
    """Send epoch/index keys to workers; prefetched keys are not completed steps."""

    def __init__(self) -> None:
        """Start with no scheduled batches."""
        self.batches: list[list[tuple[int, int]]] = []

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        """Yield exactly the current main-process batch schedule."""
        return iter(self.batches)

    def __len__(self) -> int:
        """Count scheduled batches rather than prefetched results."""
        return len(self.batches)


class WorkerPool:
    """Reuse bounded worker processes while each sample carries its immutable epoch key."""

    def __init__(self, data: DataLoader[Any]) -> None:
        """Keep the requested workers and prefetch factor, with no worker state in snapshots."""
        self.schedule = EpochBatches()
        self.loader = DataLoader(
            data.dataset,
            batch_sampler=self.schedule,
            num_workers=data.num_workers,
            generator=data.generator,
            multiprocessing_context=data.multiprocessing_context,
            prefetch_factor=data.prefetch_factor,
            persistent_workers=data.num_workers > 0,
        )

    def fetch(self, epoch: int, batches: list[list[int]]) -> Iterator[Any]:
        """Discard completed indices; a fresh process refetches only untrained keys."""
        self.schedule.batches = [[(epoch, index) for index in batch] for batch in batches]
        return iter(self.loader)
