"""Prove exact SIGTERM resume while changing prefetched data workers from four to two."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE_SECONDS = 60
POLL_SECONDS = 0.01
TERMINATED = 143
SEED = 1729
START_WORKERS = 4
PREFETCH = 2


def prove_workers(folder: Path, stops: tuple[int, ...] = (1, 6, 13)) -> list[str]:
    """Run one trainer at a time; verify actual worker counts, snapshot positions and hashes."""
    script = str(ROOT / "examples/train_workers.py")
    command = [sys.executable, "-m", "forge.runner"]
    evidence = folder / "baseline.json"
    baseline = _output(
        [
            *(*command, "--snapshot", str(folder / "unused"), "--", "python", script),
            *("--workers", "4", "--evidence", str(evidence)),
        ]
    )
    _require_workers(evidence, 4)
    lines = []
    for step in stops:
        snapshot = folder / f"stop-{step}"
        progress = folder / f"progress-{step}"
        stopped = [
            *command,
            "--snapshot",
            str(snapshot),
            "--",
            "python",
            script,
            "--workers",
            "4",
            "--progress",
            str(progress),
            "--step-delay",
            "0.1",
        ]
        with subprocess.Popen(
            stopped, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as job:
            try:
                _wait(job, progress, step)
                requested = time.monotonic()
                job.send_signal(signal.SIGTERM)
                output, errors = job.communicate(timeout=DEADLINE_SECONDS)
                elapsed = time.monotonic() - requested
                if job.returncode != TERMINATED or output:
                    raise ValueError(f"SIGTERM snapshot failed: {errors}")
                document = json.loads((snapshot / "snapshot.json").read_text())
                state = dict(document["state"]["mapping"])
                if state["step"] != step or state["index_seed"] != SEED:
                    raise ValueError("Snapshot does not describe the requested completed position.")
            finally:
                if job.poll() is None:
                    job.kill()
                    job.wait()
        resumed_evidence = folder / f"resume-{step}.json"
        resumed = _output(
            [
                *command,
                "--snapshot",
                str(folder / f"after-{step}"),
                "--resume",
                str(snapshot),
                "--",
                "python",
                script,
                "--workers",
                "2",
                "--evidence",
                str(resumed_evidence),
            ]
        )
        _require_workers(resumed_evidence, 2)
        lines.extend(
            [
                f"stop={step} workers=4->2 prefetch=2 snapshot_seconds={elapsed:.3f}",
                f"uninterrupted {baseline}",
                f"resumed       {resumed}",
            ]
        )
        if baseline != resumed:
            raise ValueError("Worker-count change altered final training weights.")
    return lines


def _output(command: list[str]) -> str:
    """Run a fresh training process to completion, with a bounded verification timeout."""
    return subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=DEADLINE_SECONDS
    ).stdout.strip()


def _require_workers(path: Path, count: int) -> None:
    """Check counts reported from inside actual worker processes, not just CLI arguments."""
    data = json.loads(path.read_text())
    if data != {"workers": [count], "prefetch": PREFETCH}:
        raise ValueError("The run did not use the expected workers with prefetch enabled.")


def _wait(job: subprocess.Popen[str], progress: Path, step: int) -> None:
    """Send the signal during a completed optimizer/scheduler step's deliberate delay."""
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if job.poll() is not None:
            raise ValueError("Training exited before the requested stop.")
        if progress.exists():
            content = progress.read_text()
            if content:
                data = json.loads(content)
                if data["step"] >= step and data["workers"] == [START_WORKERS]:
                    return
        time.sleep(POLL_SECONDS)
    raise TimeoutError("Workers did not reach the requested stop in time.")


def main() -> int:
    """Print actual multi-worker proof output and remove temporary snapshots."""
    with tempfile.TemporaryDirectory() as directory:
        for line in prove_workers(Path(directory)):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
