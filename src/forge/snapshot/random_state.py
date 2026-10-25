"""Capture the random streams used by a single-process training loop."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def capture(device: str = "cpu") -> dict[str, Any]:
    """Return Python, NumPy, torch CPU and the active GPU streams as data."""
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        raise ValueError("NumPy must use its legacy random stream for this snapshot.")
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "mps": torch.mps.get_rng_state() if device == "mps" else None,
        "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
    }


def restore(state: dict[str, Any]) -> None:
    """Restore random streams after constructing and loading training objects."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved random streams.")
        torch.cuda.set_rng_state_all(state["cuda"])
