"""Run one session of a job inside a prepared folder, on whatever machine it landed on.

The folder holds ``spec.json``, the job under ``job/``, forge's own wrapper under ``forge/``
and, when resuming, a snapshot under ``resume/``. Results go to ``out/``: ``exit_code``,
``log.txt``, ``outputs.zip`` and, when the job paused, ``snapshot/``. The same file runs on
this machine, in a Kaggle notebook and in a Modal sandbox.

Usage: python -m forge.session FOLDER [SECONDS]
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

FINISHED, PAUSED, KILLED = 0, 143, 137
SECONDS_ARGUMENT = 2
# The job is asked to save this long before the session's hard limit: a quarter of the
# session, at most a minute. Saving a snapshot needs seconds, not minutes.
SAVE_RESERVE_FRACTION = 4
SAVE_RESERVE_MAXIMUM = 60


def main() -> int:
    """Run the job once under the session limit and record how it ended."""
    folder = Path(sys.argv[1]).resolve()
    seconds = float(sys.argv[SECONDS_ARGUMENT]) if len(sys.argv) > SECONDS_ARGUMENT else None
    out = folder / "out"
    out.mkdir(exist_ok=True)
    code = _run_job(folder, out, seconds)
    _pack_outputs(folder / "job" / "outputs", out / "outputs.zip")
    _publish(out / "exit_code", f"{code}\n")
    return 0


def deadline(start: float, seconds: float) -> float:
    """When the job should save: before the hard limit by the save reserve."""
    reserve = min(seconds / SAVE_RESERVE_FRACTION, SAVE_RESERVE_MAXIMUM)
    return start + seconds - reserve


def _run_job(folder: Path, out: Path, seconds: float | None) -> int:
    """Start the wrapped job, kill it at the hard limit, and return its exit status."""
    spec = json.loads((folder / "spec.json").read_text())
    job = folder / "job"
    command = [sys.executable, "-m", "forge.runner", "--snapshot", str(out / "snapshot")]
    if (folder / "resume").is_dir():
        command += ["--resume", str(folder / "resume")]
    command += ["--", "python", str(job / spec["script"]), *spec["args"]]
    environment = dict(os.environ, PYTHONPATH=str(folder))
    start = time.time()
    if seconds is not None:
        environment["FORGE_DEADLINE"] = str(deadline(start, seconds))
    with (out / "log.txt").open("ab") as log:
        child = subprocess.Popen(
            command, cwd=job, env=environment, stdout=log, stderr=log, start_new_session=True
        )
        _publish(out / "runner.pid", f"{child.pid}\n")
        try:
            code = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            # The hard limit: stop the job and anything it started, such as data workers.
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            return KILLED
    # A pause that arrives while the job is still starting ends it before it can save.
    # Nothing is lost, so the session counts as paused and the run starts again later.
    return PAUSED if code == -signal.SIGTERM else code


def _publish(path: Path, text: str) -> None:
    """Write a small file in one rename, so a reader never sees it half-written."""
    pending = path.with_name(path.name + ".pending")
    pending.write_text(text)
    pending.replace(path)


def _pack_outputs(outputs: Path, archive: Path) -> None:
    """Zip whatever the job wrote under ``outputs/``, so it can travel as one file."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        if outputs.is_dir():
            for path in sorted(outputs.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(outputs))


if __name__ == "__main__":
    sys.exit(main())
