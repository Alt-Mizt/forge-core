"""Compare every training loss and final weights after a CPU/MPS move."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

STEPS = 24
STOP_STEP = 6


def main() -> int:
    """Print actual losses, backend evidence and largest measured differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("moved", type=Path)
    parser.add_argument("--source", choices=("cpu", "mps"), default="cpu")
    options = parser.parse_args()
    baseline = options.baseline
    moved = options.moved
    source = options.source
    destination = "mps" if source == "cpu" else "cpu"
    unmoved = json.loads((baseline / "losses.json").read_text())
    continued = json.loads((moved / "losses.json").read_text())
    if len(unmoved["losses"]) != STEPS or len(continued["losses"]) != STEPS:
        raise ValueError("Both runs must contain every completed training step.")
    if unmoved["devices"] != [source] * STEPS:
        raise ValueError("The baseline must remain on the source device throughout.")
    if continued["devices"] != [source] * STOP_STEP + [destination] * (STEPS - STOP_STEP):
        raise ValueError("The moved run must actually change device after the stop.")
    if unmoved["losses"][:STOP_STEP] != continued["losses"][:STOP_STEP]:
        raise ValueError("The source prefix changed before the hardware move.")
    if not all(math.isfinite(value) for value in unmoved["losses"] + continued["losses"]):
        raise ValueError("Loss comparison requires finite values at every step.")
    differences = []
    print(f"step {source}_loss moved_loss difference device")
    for step, (baseline_loss, moved_loss) in enumerate(
        zip(unmoved["losses"], continued["losses"], strict=True), 1
    ):
        difference = abs(baseline_loss - moved_loss)
        differences.append(difference)
        print(
            f"{step:02d} {baseline_loss:.9f} {moved_loss:.9f} "
            f"{difference:.9g} {continued['devices'][step - 1]}"
        )
    baseline_weights = load_file(baseline / "weights.safetensors")
    moved_weights = load_file(moved / "weights.safetensors")
    if baseline_weights.keys() != moved_weights.keys():
        raise ValueError("Both runs must finish with the same model structure.")
    if not all(
        torch.isfinite(value).all()
        for value in [*baseline_weights.values(), *moved_weights.values()]
    ):
        raise ValueError("Weight comparison requires finite parameters.")
    weight_difference = max(
        float((baseline_weights[name] - moved_weights[name]).abs().max())
        for name in baseline_weights
    )
    print(f"largest_loss_difference {max(differences):.9g}")
    print(f"{source}_sha256 {unmoved['sha256']}")
    print(f"moved_sha256 {continued['sha256']}")
    print(f"largest_weight_difference {weight_difference:.9g}")
    print(
        f"parameters {unmoved['parameters']}; {source} steps 1-{STOP_STEP}, "
        f"{destination} steps {STOP_STEP + 1}-{STEPS}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
