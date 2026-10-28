# Examples

All on generated data, small enough for a laptop CPU.

- `tiny/train.py`: a plain PyTorch loop with no forge import. It prints the SHA-256 of
  its final weights and writes it to `outputs/weights-sha256.txt`. `--step-delay` slows
  each step so that a run spans several short sessions. Used by `scripts/demo-hop.sh`
  and `scripts/demo-wrapper.sh`.
- `train_tiny.py`: the same loop using `forge.resumable` directly. Used by
  `scripts/demo-resume.sh`.
- `train_workers.py`: augmented waveforms loaded by four data workers, keyed by
  `(epoch, index)` so a resume with two workers sees the same data.
- `train_migrate.py`: a CPU, MPS or CUDA run that records every loss and device, for
  `scripts/demo-cpu-to-gpu.sh`.
