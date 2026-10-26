"""Prove SIGTERM snapshot latency and fresh-process resume for an unmodified loop."""

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
STARTUP_SECONDS = 90
POLL_SECONDS = 0.01
TERMINATED = 143


def prove_wrapper(folder: Path) -> list[str]:
    """Run training processes one at a time and return their actual matching evidence."""
    script = str(ROOT / "examples/tiny/train.py")
    wrapper = [sys.executable, "-m", "forge.runner"]
    straight = _output([*wrapper, "--snapshot", str(folder / "unused"), "--", "python", script])
    ordinary = _output([sys.executable, script])
    if ordinary != straight:
        raise ValueError("Wrapping changed the training script's final weights.")
    evidence: list[str] = []
    for step in (1, 6, 13):
        snapshot = folder / f"snapshot-{step}"
        progress = folder / f"progress-{step}"
        stopped = [
            *wrapper,
            "--snapshot",
            str(snapshot),
            "--",
            "python",
            script,
            "--step-delay",
            "0.1",
            "--progress",
            str(progress),
        ]
        with subprocess.Popen(
            stopped, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=folder
        ) as job:
            try:
                _wait_for_step(job, progress, step)
                requested = time.monotonic()
                job.send_signal(signal.SIGTERM)
                output, errors = job.communicate(timeout=DEADLINE_SECONDS)
                elapsed = time.monotonic() - requested
                if (
                    job.returncode != TERMINATED
                    or output
                    or not (snapshot / "snapshot.json").is_file()
                ):
                    raise ValueError(f"SIGTERM did not publish a snapshot: {errors}")
                document = json.loads((snapshot / "snapshot.json").read_text())
                state = dict(document["state"]["mapping"])
                if state["step"] != step:
                    raise ValueError("SIGTERM did not stop at the requested completed step.")
            finally:
                if job.poll() is None:
                    job.kill()
                    job.wait()
        resumed = _output(
            [
                *wrapper,
                *("--snapshot", str(folder / f"after-{step}"), "--resume", str(snapshot)),
                *("--", "python", script),
            ]
        )
        evidence.extend(
            [
                f"stop={step} snapshot_seconds={elapsed:.3f}",
                f"uninterrupted {straight}",
                f"resumed       {resumed}",
            ]
        )
        if resumed != straight:
            raise ValueError("Fresh-process SIGTERM resume changed the final weights.")
    return evidence


def _output(command: list[str]) -> str:
    """Run one process to completion without sharing a training process with another."""
    with tempfile.TemporaryDirectory() as directory:
        # The example writes outputs/ in its working folder; keep that out of the repo.
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=STARTUP_SECONDS,
            cwd=directory,
        ).stdout.strip()


def _wait_for_step(job: subprocess.Popen[str], progress: Path, step: int) -> None:
    """Wait until user updates complete, then send SIGTERM during its deliberate delay."""
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if job.poll() is not None:
            raise ValueError("The training process exited before the requested step.")
        if progress.exists():
            content = progress.read_text()
            if content and int(content) >= step:
                return
        time.sleep(POLL_SECONDS)
    raise TimeoutError("Training did not reach the requested step in time.")


def main() -> int:
    """Print evidence for the wrapper demo, keeping snapshot files outside the repo."""
    with tempfile.TemporaryDirectory() as directory:
        for line in prove_wrapper(Path(directory)):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
