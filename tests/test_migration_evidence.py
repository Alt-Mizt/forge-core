"""Invalid numbers cannot be reported as a successful hardware comparison."""

import json
import sys

import pytest
from compare_migration import main


@pytest.mark.parametrize("source,destination", [("cpu", "mps"), ("mps", "cpu")])
def test_nan_loss_is_refused_instead_of_disappearing_from_the_largest_difference(
    tmp_path, monkeypatch, source, destination
):
    cpu = tmp_path / "cpu"
    moved = tmp_path / "moved"
    cpu.mkdir()
    moved.mkdir()
    (cpu / "losses.json").write_text(json.dumps({"losses": [1.0] * 24, "devices": [source] * 24}))
    losses = [1.0] * 24
    losses[7] = float("nan")
    (moved / "losses.json").write_text(
        json.dumps({"losses": losses, "devices": [source] * 6 + [destination] * 18})
    )
    monkeypatch.setattr(sys, "argv", ["compare", str(cpu), str(moved), "--source", source])
    with pytest.raises(ValueError, match="finite"):
        main()
