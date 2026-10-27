"""Train random augmented synthetic waveforms using independent epoch/index seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

SEED = 1729
SAMPLES = 48
WIDTH = 64
BATCH_SIZE = 8
EPOCHS = 4
MASK_WIDTH = 4


class AudioClips(Dataset):
    """Index-addressable generated clips; no augmentation consumes a worker RNG."""

    resumable_by_index = True

    def __init__(self, seed: int) -> None:
        """Generate fixed base waveforms, using a local generator."""
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        phase = torch.linspace(0, 1, WIDTH)
        frequencies = torch.arange(1, SAMPLES + 1).remainder(7) + 1
        self.clips = torch.sin(2 * torch.pi * frequencies[:, None] * phase)
        self.clips += 0.01 * torch.randn(SAMPLES, WIDTH, generator=generator)

    def __len__(self) -> int:
        """Return the immutable number of clip indices."""
        return SAMPLES

    def __getitem__(self, key: tuple[int, int] | int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply gain, Gaussian noise and time masking from the epoch/index key."""
        epoch, index = key if isinstance(key, tuple) else (0, key)
        digest = hashlib.sha256(f"{self.seed}:{epoch}:{index}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        generator = torch.Generator().manual_seed(seed)
        numpy_generator = np.random.default_rng(seed)
        python_generator = random.Random(seed)
        clip = self.clips[index].clone()
        target = clip.square().mean().reshape(1)
        clip *= float(numpy_generator.uniform(0.8, 1.2))
        clip += 0.02 * torch.randn(WIDTH, generator=generator)
        start = python_generator.randrange(WIDTH - MASK_WIDTH)
        clip[start : start + MASK_WIDTH] = 0
        worker = get_worker_info()
        return clip, target, worker.num_workers if worker else 0


def main() -> int:
    """Train a plain loop; the wrapper supplies epoch/index keys and SIGTERM resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--step-delay", type=float, default=0)
    options = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 16), torch.nn.Dropout(0.25), torch.nn.Linear(16, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    data = DataLoader(
        AudioClips(SEED),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=options.workers,
        prefetch_factor=2,
        multiprocessing_context="spawn",
        generator=torch.Generator().manual_seed(SEED),
    )
    step = 0
    observed: set[int] = set()
    for _epoch in range(EPOCHS):
        for clips, labels, workers in data:
            observed.update(workers.tolist())
            if observed != {options.workers}:
                raise ValueError("Actual data workers differ from the requested worker count.")
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(clips), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if options.progress:
                options.progress.write_text(json.dumps({"step": step, "workers": sorted(observed)}))
            time.sleep(options.step_delay)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    if options.evidence:
        options.evidence.write_text(
            json.dumps({"workers": sorted(observed), "prefetch": data.prefetch_factor})
        )
    sys.stdout.write(digest.hexdigest() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
