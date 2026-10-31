"""A job: a training script, the folder around it, an optional requirements file, arguments.

The folder is packed into one zip archive that travels to every session. Caches, virtual
environments and earlier outputs are left out, and an oversized folder is refused.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

MAX_JOB_BYTES = 50 * 1024 * 1024
MEBIBYTE = 1024 * 1024
EXCLUDED_FOLDERS = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".forge", "outputs"}
)
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
EXCLUDED_NAMES = frozenset({".DS_Store"})
TORCH_PIN = re.compile(r"^\s*torch\s*==\s*([0-9][0-9A-Za-z.]*)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Job:
    """What to run: ``script`` is relative to ``folder``; ``requirements`` are pip lines."""

    folder: Path
    script: str
    args: list[str]
    requirements: list[str]

    @property
    def torch_pin(self) -> str | None:
        """The torch version the requirements pin with ``torch==``, if any."""
        match = TORCH_PIN.search("\n".join(self.requirements))
        return match.group(1) if match else None


def describe(script: Path, args: list[str], requirements: Path | None = None) -> Job:
    """Describe a job from its script; ``requirements.txt`` beside it is used if present."""
    script = script.resolve()
    if not script.is_file():
        raise ValueError(f"The training script {script.name} does not exist.")
    folder = script.parent
    requirements = requirements or folder / "requirements.txt"
    lines = requirements.read_text().splitlines() if requirements.is_file() else []
    wanted = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return Job(folder, script.name, args, wanted)


def pack(job: Job, archive: Path) -> str:
    """Zip the job folder into ``archive`` and return the archive's SHA-256.

    Raises:
        ValueError: When the packed files exceed the size limit.
    """
    files = list(_files(job.folder))
    size = sum(path.stat().st_size for path in files)
    if size > MAX_JOB_BYTES:
        raise ValueError(
            f"The job folder {job.folder.name} holds {size // MEBIBYTE} MiB; forge sends at most "
            f"{MAX_JOB_BYTES // MEBIBYTE} MiB. Move data out of the folder and download it in "
            "the script instead."
        )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(job.folder))
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def _files(folder: Path) -> list[Path]:
    """Every file in the job folder that should travel, in a stable order."""
    chosen = []
    for path in sorted(folder.rglob("*")):
        parts = path.relative_to(folder).parts
        if EXCLUDED_FOLDERS.intersection(parts[:-1]) or not path.is_file():
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
            continue
        chosen.append(path)
    return chosen


def session_torch(job: Job, snapshot: Path | None, installed: str) -> str:
    """The torch version the next session must use, so its snapshot can be restored.

    A snapshot only resumes under the torch version that wrote it. Without a snapshot, the
    job's pin wins, then the version installed here.

    Raises:
        ValueError: When the job pins a different torch than the snapshot was written with.
    """
    written = snapshot_torch(snapshot) if snapshot is not None else None
    pinned = job.torch_pin
    if written and pinned and written != pinned:
        raise ValueError(
            f"The snapshot was written with torch {written}, but the job pins torch {pinned}. "
            f"Pin torch=={written} in the requirements to continue this run."
        )
    return written or pinned or plain_version(installed)


def snapshot_torch(snapshot: Path) -> str:
    """Read the torch version a snapshot records, without loading torch."""
    document = json.loads((snapshot / "snapshot.json").read_text())
    for key, value in document["state"]["mapping"]:
        if key == "torch_version":
            return plain_version(value)
    raise ValueError("The snapshot does not record its torch version.")


def plain_version(version: str) -> str:
    """A torch version without its local build tag: ``2.8.0+cu128`` becomes ``2.8.0``."""
    return version.split("+", 1)[0]
