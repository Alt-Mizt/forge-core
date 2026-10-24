"""Write forge's own files so that a crash, or a second forge, cannot leave them half-written.

A writer replaces a file in one rename or appends one synced line, and holds a lock file
while it does. A reader never sees a partial file.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(lock: Path) -> Iterator[None]:
    """Hold ``lock`` for the duration, so one process at a time does what it guards."""
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_atomically(path: Path, data: bytes) -> None:
    """Write a pending copy, sync it, then rename it over ``path`` in one step."""
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    sync_directory(path.parent)


def append_synced(path: Path, data: bytes) -> None:
    """Append ``data`` to ``path`` and sync it to disk before returning."""
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(folder: Path) -> None:
    """Sync a directory's entries, so a file created in it survives a crash."""
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
