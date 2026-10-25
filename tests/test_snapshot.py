"""Fresh processes preserve training state; damaged snapshots fail closed."""

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as load_tensors
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from forge.snapshot import codec, load, resumable, save

ROOT = Path(__file__).resolve().parents[1]


def train(*arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/train_tiny.py"), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_fresh_process_resume_matches_at_three_positions_including_epoch_end(tmp_path):
    straight = train()
    for step in (1, 6, 13):
        folder = str(tmp_path / str(step))
        train("--stop-at", str(step), "--snapshot", folder)
        assert train("--resume", folder) == straight


def test_tensor_and_scalar_state_round_trip_without_pickle(tmp_path):
    folder = tmp_path / "snapshot"
    state = {"optimizer": {0: {"momentum": torch.arange(4)}}, "rng": (1, [2, 3])}
    save(state, folder)
    restored = load(folder)
    assert torch.equal(restored["optimizer"][0]["momentum"], state["optimizer"][0]["momentum"])
    assert restored["rng"] == state["rng"]
    assert {path.name for path in folder.iterdir()} == {"snapshot.json", "tensors.safetensors"}


@pytest.mark.parametrize("name", ["snapshot.json", "tensors.safetensors"])
def test_corrupt_state_is_rejected_before_it_can_replace_a_run(tmp_path, name):
    folder = tmp_path / "snapshot"
    save({"step": 2, "weight": torch.ones(2)}, folder)
    path = folder / name
    if name == "snapshot.json":
        document = json.loads(path.read_text())
        document["state"]["mapping"][0][1] = 99
        path.write_text(json.dumps(document))
    else:
        path.write_bytes(path.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="integrity"):
        load(folder)


def test_existing_snapshot_is_preserved_when_another_save_is_attempted(tmp_path):
    folder = tmp_path / "snapshot"
    save({"step": 2}, folder)
    before = hashlib.sha256((folder / "snapshot.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        save({"step": 3}, folder)
    assert hashlib.sha256((folder / "snapshot.json").read_bytes()).hexdigest() == before


def test_publication_syncs_directory_entries_before_acknowledging_success(tmp_path, monkeypatch):
    events = []
    original_sync = os.fsync
    original_rename = Path.rename

    def sync(descriptor):
        events.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        original_sync(descriptor)

    def rename(path, destination):
        events.append("rename")
        return original_rename(path, destination)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(Path, "rename", rename)
    save({"weight": torch.ones(2)}, tmp_path / "snapshot")
    assert events == ["file", "file", "directory", "rename", "directory"]


def test_deserialization_uses_the_verified_bytes_even_if_the_file_changes(tmp_path, monkeypatch):
    folder = tmp_path / "snapshot"
    save({"weight": torch.ones(2)}, folder)
    original_load = load_tensors

    def replace_file_then_load(content):
        (folder / "tensors.safetensors").write_bytes(b"replaced")
        return original_load(content)

    monkeypatch.setattr(codec, "load_tensors", replace_file_then_load)
    assert torch.equal(load(folder)["weight"], torch.ones(2))


def test_tensor_hashing_and_loading_never_allocate_the_entire_file_as_bytes(tmp_path, monkeypatch):
    original = Path.read_bytes

    def refuse_whole_tensor_read(path):
        if path.name == "tensors.safetensors":
            raise AssertionError("Tensor files must be streamed.")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_whole_tensor_read)
    folder = tmp_path / "snapshot"
    save({"weight": torch.arange(4)}, folder)
    restored = load(folder)
    assert torch.equal(restored["weight"], torch.arange(4))
    assert not list(tmp_path.glob(".verified-*"))


def test_a_partial_sampler_is_refused_instead_of_changing_the_requested_training():
    generator = torch.Generator().manual_seed(1)
    dataset = TensorDataset(torch.ones(4, 1))
    data = DataLoader(
        dataset,
        batch_size=1,
        generator=generator,
        sampler=RandomSampler(dataset, generator=generator, num_samples=2),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="complete permutation"):
        resumable(model, optimizer, data)


def test_missing_snapshot_parent_is_refused_before_publication(tmp_path):
    with pytest.raises(FileNotFoundError, match="parent folder"):
        save({"step": 2}, tmp_path / "missing" / "snapshot")
    assert not (tmp_path / "missing").exists()


class GeneratorDataset(torch.utils.data.Dataset):
    """Expose loader-generator ordering through randomly generated input values."""

    def __init__(self, generator):
        self.generator = generator

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return index, torch.rand((), generator=self.generator)


def test_partial_batch_generator_timing_matches_plain_loader_and_resume(tmp_path):
    def objects():
        generator = torch.Generator().manual_seed(1)
        data = DataLoader(
            GeneratorDataset(generator), batch_size=2, shuffle=True, generator=generator
        )
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        return data, resumable(model, optimizer, data)

    plain, _unused = objects()
    expected = [
        (indices.tolist(), values.tolist()) for _epoch in range(2) for indices, values in plain
    ]
    _data, state = objects()
    observed = []
    for indices, values in state:
        observed.append((indices.tolist(), values.tolist()))
    state.save(tmp_path / "partial-end")
    _data, resumed = objects()
    resumed.restore(tmp_path / "partial-end")
    for indices, values in resumed:
        observed.append((indices.tolist(), values.tolist()))
    assert observed == expected


def test_a_snapshot_at_the_final_partial_batch_does_not_repeat_sampler_exhaustion(tmp_path):
    def objects():
        generator = torch.Generator().manual_seed(1)
        data = DataLoader(
            GeneratorDataset(generator), batch_size=2, shuffle=True, generator=generator
        )
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        return resumable(model, optimizer, data)

    state = objects()
    batches = iter(state)
    for _step in range(3):
        next(batches)
    state.save(tmp_path / "partial-boundary")
    assert load(tmp_path / "partial-boundary")["sampler_exhausted"] is True
    resumed = objects()
    resumed.restore(tmp_path / "partial-boundary")
    assert list(resumed) == []
    next_epoch = [(i.tolist(), v.tolist()) for i, v in resumed]
    list(batches)
    expected = [(i.tolist(), v.tolist()) for i, v in state]
    assert next_epoch == expected
