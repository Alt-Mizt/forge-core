"""CUDA stream state is safe tensor data and cannot silently lose a device."""

import pytest
import torch

from forge.snapshot import load, random_state, save


def test_cuda_random_streams_round_trip_without_a_gpu_or_pickle(tmp_path, monkeypatch):
    expected = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: expected)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    restored = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.extend)
    state = random_state.capture("cuda")
    save(state, tmp_path / "streams")
    random_state.restore(load(tmp_path / "streams"))
    assert len(restored) == 1
    assert torch.equal(restored[0], expected[0])


def test_cuda_device_count_mismatch_refuses_instead_of_leaving_random_streams_unset(
    monkeypatch,
):
    state = random_state.capture()
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="device count"):
        random_state.restore(state)
