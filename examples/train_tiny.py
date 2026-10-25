"""Train generated data with shuffle, dropout, AdamW and a learning-rate schedule."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from forge.snapshot import resumable

SEED = 1729
SAMPLES = 48
BATCH_SIZE = 8
EPOCHS = 4


def main() -> int:
    """Train through or stop after a completed batch, emitting a final weight hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    options = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(mode=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    features = torch.randn(SAMPLES, 4)
    targets = features.sum(dim=1, keepdim=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.Dropout(0.25), torch.nn.Linear(16, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    data = DataLoader(
        TensorDataset(features, targets),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    run = resumable(model, optimizer, data, scheduler)
    if options.resume:
        run.restore(options.resume)
    while run.epoch < EPOCHS:
        for inputs, labels in run:
            optimizer.zero_grad()
            jitter = random.random() * 0.001 + float(np.random.random()) * 0.001
            loss = torch.nn.functional.mse_loss(model(inputs + jitter), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if run.step == options.stop_at:
                if options.snapshot is None:
                    parser.error("--stop-at requires --snapshot")
                run.save(options.snapshot)
                return 0
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    sys.stdout.write(digest.hexdigest() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
