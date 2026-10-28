"""Measure hardware differences while continuing the same tiny CPU training state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from forge.snapshot import load, resumable, save

SEED = 1729
SAMPLES = 48
BATCH_SIZE = 8
EPOCHS = 4


def main() -> int:
    """Train on CPU, MPS or CUDA, preserving completed losses across a process move."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    options = parser.parse_args()
    _require_device(options.device, parser)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(mode=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    features = torch.randn(SAMPLES, 4)
    targets = features.sum(dim=1, keepdim=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(
        options.device
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
    losses: list[float] = []
    devices: list[str] = []
    if options.resume:
        document = load(options.resume)
        run.restore_state(document)
        losses = document["losses"]
        devices = document["devices"]
    while run.epoch < EPOCHS:
        for inputs, labels in run:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(
                model(inputs.to(options.device)), labels.to(options.device)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            devices.append(next(model.parameters()).device.type)
            if run.step == options.stop_at:
                if options.snapshot is None:
                    parser.error("--stop-at requires --snapshot")
                document = run.state_dict()
                document.update({"losses": losses, "devices": devices})
                save(document, options.snapshot)
                return 0
    weights = {
        name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
    }
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    options.output.mkdir()
    save_file(weights, options.output / "weights.safetensors")
    (options.output / "losses.json").write_text(
        json.dumps(
            {
                "losses": losses,
                "devices": devices,
                "sha256": digest.hexdigest(),
                "parameters": sum(value.numel() for value in model.parameters()),
                "torch_version": torch.__version__,
                "gpu": (
                    torch.cuda.get_device_name(0) if options.device == "cuda" else options.device
                ),
                "gpu_count": torch.cuda.device_count() if options.device == "cuda" else 0,
            }
        )
    )
    return 0


def _require_device(device: str, parser: argparse.ArgumentParser) -> None:
    """Refuse absent GPU backends before creating a model or snapshot."""
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Run this demo with access to the Mac GPU.")
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. This job requires an allocated NVIDIA GPU.")


if __name__ == "__main__":
    sys.exit(main())
