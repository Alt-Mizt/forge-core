"""Native Mac GPU snapshots preserve optimizer moments and dropout random state."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from forge.snapshot import resumable


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Apple GPU (MPS)")
def test_mps_snapshot_preserves_the_next_update_and_adamw_moments(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(1729)

    def objects():
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.25)).to("mps")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        data = DataLoader(
            TensorDataset(torch.ones(4, 4)),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        return model, optimizer, resumable(model, optimizer, data)

    model, optimizer, state = objects()
    inputs = next(iter(state))[0].to("mps")
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    state.save(tmp_path / "gpu-state")
    saved_moments = [
        {name: values[name].detach().cpu().clone() for name in ("step", "exp_avg", "exp_avg_sq")}
        for values in optimizer.state.values()
    ]
    expected = model(inputs)
    expected.sum().backward()
    optimizer.step()
    restored_model, restored_optimizer, restored = objects()
    restored.restore(tmp_path / "gpu-state")
    for saved, values in zip(saved_moments, restored_optimizer.state.values(), strict=True):
        for name, expected_moment in saved.items():
            assert torch.equal(values[name].detach().cpu(), expected_moment)
    actual = restored_model(inputs)
    assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    actual.sum().backward()
    restored_optimizer.step()
    for original, resumed in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(original.detach().cpu(), resumed.detach().cpu())
    for original, resumed in zip(
        optimizer.state.values(), restored_optimizer.state.values(), strict=True
    ):
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(original[name].detach().cpu(), resumed[name].detach().cpu())
    for values in restored_optimizer.state.values():
        assert values["exp_avg"].device.type == "mps"
        assert values["exp_avg_sq"].device.type == "mps"
    assert restored.step == 1
