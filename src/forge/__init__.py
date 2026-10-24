"""forge runs a training job on whatever GPU time your accounts have, moving it by snapshot.

The package root only re-exports :func:`resumable`, imported lazily so the CLI starts without torch.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from forge.snapshot import Resumable


def resumable(
    model: torch.nn.Module,
    optimizer: Optimizer,
    data: DataLoader[Any],
    scheduler: LRScheduler | None = None,
) -> Resumable:
    """Bind a training loop's objects so it can save and resume; see forge.snapshot."""
    return import_module("forge.snapshot").resumable(model, optimizer, data, scheduler)
