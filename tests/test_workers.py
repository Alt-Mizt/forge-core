"""Changed workers cannot change indexed augmentation or silently accept worker RNG."""

from pathlib import Path
from runpy import run_path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from workers_proof import prove_workers

from forge.snapshot import resumable

ROOT = Path(__file__).resolve().parents[1]
AUDIO_CLIPS = run_path(str(ROOT / "examples/train_workers.py"), run_name="sample_contract")[
    "AudioClips"
]


def test_four_to_two_workers_resume_matches_in_a_fresh_process_at_epoch_end(tmp_path):
    lines = prove_workers(tmp_path, stops=(6,))
    assert len(lines) == 3
    assert lines[1].split()[-1] == lines[2].split()[-1]
    assert "workers=4->2 prefetch=2" in lines[0]


def test_augmentation_is_identical_when_refetched_and_changes_with_the_epoch():
    data = AUDIO_CLIPS(1729)
    clip, target, _workers = data[(2, 17)]
    other_clip, other_target, _workers = data[(2, 17)]
    next_clip, next_target, _workers = data[(3, 17)]
    assert torch.equal(clip, other_clip) and torch.equal(target, other_target)
    assert not torch.equal(clip, next_clip)
    assert torch.equal(target, next_target)


def test_worker_rng_without_an_indexed_contract_is_refused_before_workers_start():
    data = DataLoader(
        TensorDataset(torch.ones(4, 1)),
        batch_size=2,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator().manual_seed(1),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="keyed by"):
        resumable(model, optimizer, data)


def test_changed_augmentation_seed_refuses_before_loading_weights(tmp_path):
    def objects(seed):
        model = torch.nn.Linear(1, 1)
        data = DataLoader(
            AUDIO_CLIPS(seed),
            batch_size=8,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        return model, resumable(model, torch.optim.AdamW(model.parameters()), data)

    _model, original = objects(1)
    original.save(tmp_path / "state")
    model, resumed = objects(2)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="augmentation seed"):
        resumed.restore(tmp_path / "state")
    assert torch.equal(model.weight.detach(), before)
