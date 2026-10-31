"""Build the folder a session runs in, as one zip file that any provider can carry.

Layout: ``spec.json``, ``job/`` (the packed job), ``forge/`` (forge's wrapper, session
runner and snapshot code, nothing else) and, when resuming, ``resume/`` (the snapshot).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import forge

FORGE_FILES = ("__init__.py", "runner.py", "session.py")


@dataclass(frozen=True)
class SessionSpec:
    """Everything a provider needs to start one session of a run.

    Attributes:
        run: The run's name.
        number: The session's number within the run, from 1.
        archive: The packed job (see ``forge.jobs.pack``).
        script: The training script, relative to the job folder.
        args: Arguments for the script.
        requirements: Extra pip requirements for a cloud session.
        resume: The snapshot to continue from, or None for the first session.
        seconds: The session's hard limit, or None for no limit.
        torch: The torch version the session must run.
    """

    run: str
    number: int
    archive: Path
    script: str
    args: list[str]
    requirements: list[str]
    resume: Path | None
    seconds: int | None
    torch: str

    @property
    def handle(self) -> str:
        """The session's name at the provider, known before it starts."""
        return f"forge-{self.run}-{self.number}"


def build(spec: SessionSpec) -> bytes:
    """Return the session folder as zip bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("spec.json", json.dumps({"script": spec.script, "args": spec.args}))
        with zipfile.ZipFile(spec.archive) as job:
            for name in job.namelist():
                bundle.writestr(f"job/{name}", job.read(name))
        for path in _forge_sources():
            bundle.write(path, Path("forge") / path.relative_to(_package()))
        if spec.resume is not None:
            for path in sorted(spec.resume.iterdir()):
                bundle.write(path, f"resume/{path.name}")
    return buffer.getvalue()


def unpack(content: bytes, folder: Path) -> None:
    """Extract a session bundle into an empty folder."""
    folder.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        bundle.extractall(folder)


def _package() -> Path:
    """The installed forge package folder."""
    return Path(forge.__file__).parent


def _forge_sources() -> list[Path]:
    """The parts of forge a session needs: the wrapper, the session runner and snapshots."""
    package = _package()
    paths = [package / name for name in FORGE_FILES]
    return paths + sorted((package / "snapshot").glob("*.py"))
