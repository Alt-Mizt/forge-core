"""A job: a training script, the folder around it, an optional requirements file, arguments.

The folder is packed into one zip archive that travels to every session. Caches, virtual
environments and earlier outputs are left out, and an oversized folder is refused. A
``forge.toml`` beside the script may declare what the job needs:

    [job]
    gpu = "a100-80gb"   # the same as --gpu
    memory_gb = 24      # the least GPU memory that fits; smaller GPUs are passed over
    hours = 6           # how long the job runs, for forge plan
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_JOB_BYTES = 50 * 1024 * 1024
MEBIBYTE = 1024 * 1024
EXCLUDED_FOLDERS = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".forge", "outputs"}
)
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
EXCLUDED_NAMES = frozenset({".DS_Store"})
DECLARATION = "forge.toml"
DECLARED_KEYS = ("gpu", "memory_gb", "hours")
TORCH_PIN = re.compile(r"^\s*torch\s*==\s*([0-9][0-9A-Za-z.]*)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Declaration:
    """What a job's ``forge.toml`` says it needs; each value is None when it says nothing."""

    gpu: str | None = None
    memory_gb: int | None = None
    hours: float | None = None


@dataclass(frozen=True)
class Job:
    """What to run: ``script`` is relative to ``folder``; ``requirements`` are pip lines."""

    folder: Path
    script: str
    args: list[str]
    requirements: list[str]
    declaration: Declaration = field(default_factory=Declaration)

    @property
    def torch_pin(self) -> str | None:
        """The torch version the requirements pin with ``torch==``, if any."""
        match = TORCH_PIN.search("\n".join(self.requirements))
        return match.group(1) if match else None


def describe(script: Path, args: list[str], requirements: Path | None = None) -> Job:
    """Describe a job from its script; ``requirements.txt`` and ``forge.toml`` beside it count."""
    script = script.resolve()
    if not script.is_file():
        raise ValueError(f"The training script {script.name} does not exist.")
    folder = script.parent
    requirements = requirements or folder / "requirements.txt"
    lines = requirements.read_text().splitlines() if requirements.is_file() else []
    wanted = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return Job(folder, script.name, args, wanted, declared(folder / DECLARATION))


def declared(path: Path) -> Declaration:
    """Read a job's ``forge.toml``; a missing file declares nothing.

    Raises:
        ValueError: For a key forge does not know or a value of the wrong kind, naming it.
    """
    if not path.is_file():
        return Declaration()
    document = tomllib.loads(path.read_text())
    for key in document:
        if key != "job":
            raise ValueError(f"{path.name} has an unknown table or key {key!r}; use [job].")
    table: dict[str, Any] = document.get("job", {})
    for key in table:
        if key not in DECLARED_KEYS:
            known = ", ".join(DECLARED_KEYS)
            raise ValueError(f"{path.name} has an unknown key {key!r} in [job]. Use: {known}.")
    return _declaration(path, table)


def _declaration(path: Path, table: dict[str, Any]) -> Declaration:
    """The declaration in a ``[job]`` table, refusing a value of the wrong kind by its key."""
    gpu, memory_gb, hours = (table.get(key) for key in DECLARED_KEYS)
    if gpu is not None and not isinstance(gpu, str):
        raise ValueError(f'{path.name}: gpu must be a GPU name, such as gpu = "T4".')
    if memory_gb is not None and not (_positive(memory_gb) and isinstance(memory_gb, int)):
        raise ValueError(f"{path.name}: memory_gb must be a whole number of GB, such as 24.")
    if hours is not None and not _positive(hours):
        raise ValueError(f"{path.name}: hours must be a number above 0, such as 6.")
    return Declaration(gpu, memory_gb, float(hours) if hours is not None else None)


def _positive(value: object) -> bool:
    """Whether a TOML value is a number above zero (TOML's true and false are not)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


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
