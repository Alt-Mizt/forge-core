"""Publish and verify tensor snapshots without executable deserialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file as load_tensors
from safetensors.torch import save_file

TENSOR_FILE = "tensors.safetensors"
FORMAT_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024


def save(run_state: dict[str, Any], folder: str | Path) -> None:
    """Publish a new snapshot folder atomically; refuse to replace an existing one.

    JSON's own digest covers its canonical state payload, excluding the digest table.
    Tensor hashes cover the exact bytes on disk. Readers verify both before decoding.
    """
    destination = Path(folder)
    if not destination.parent.is_dir():
        raise FileNotFoundError("Create the snapshot parent folder before saving.")
    if destination.exists():
        raise FileExistsError(f"Snapshot {destination} already exists; choose a new folder.")
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
    try:
        tensors: dict[str, torch.Tensor] = {}
        payload = {"format": FORMAT_VERSION, "state": _encode(run_state, tensors)}
        save_file(tensors, staging / TENSOR_FILE)
        document = dict(payload)
        document["sha256"] = {
            TENSOR_FILE: _file_digest(staging / TENSOR_FILE),
            "snapshot.json": _digest(_canonical(payload)),
        }
        (staging / "snapshot.json").write_bytes(_canonical(document))
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(staging)
        staging.rename(destination)
        _sync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load(folder: str | Path) -> dict[str, Any]:
    """Verify a snapshot and return its state using only JSON and safetensors."""
    source = Path(folder)
    document = json.loads((source / "snapshot.json").read_bytes())
    payload = {"format": document["format"], "state": document["state"]}
    expected = document["sha256"]
    if payload["format"] != FORMAT_VERSION:
        raise ValueError("Unsupported snapshot format.")
    if expected.get("snapshot.json") != _digest(_canonical(payload)) or set(expected) != {
        TENSOR_FILE,
        "snapshot.json",
    }:
        raise ValueError("Snapshot integrity check failed; use an intact snapshot.")
    # Decode the verified private copy, not a source reopened after verification.
    # The anonymous mapping survives unlink and cannot follow source replacement.
    with tempfile.TemporaryDirectory(prefix="forge-verified-") as directory:
        copied = Path(directory) / TENSOR_FILE
        digest = hashlib.sha256()
        with (source / TENSOR_FILE).open("rb") as original, copied.open("xb") as destination:
            while chunk := original.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
                destination.write(chunk)
        if expected[TENSOR_FILE] != digest.hexdigest():
            raise ValueError("Snapshot integrity check failed; use an intact snapshot.")
        state = _decode(payload["state"], load_tensors(str(copied)))
    if not isinstance(state, dict):
        raise ValueError("Snapshot state must be a mapping.")
    return cast(dict[str, Any], state)


def _encode(value: object, tensors: dict[str, torch.Tensor]) -> object:
    """Encode nested optimizer state while preserving integer keys and tuples."""
    if isinstance(value, torch.Tensor):
        name = str(len(tensors))
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {"tensor": name}
    if isinstance(value, dict):
        return {"mapping": [[_encode(k, tensors), _encode(v, tensors)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item, tensors) for item in value]}
    if isinstance(value, list):
        return {"list": [_encode(item, tensors) for item in value]}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"Snapshot cannot encode state of type {type(value).__name__}.")


def _decode(value: object, tensors: dict[str, torch.Tensor]) -> object:
    """Decode only the closed set of data tags written by this module."""
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValueError("Invalid snapshot state tag.")
    tag, items = next(iter(value.items()))
    if tag == "tensor":
        if not isinstance(items, str):
            raise ValueError("Invalid tensor reference.")
        return tensors[items]
    if tag == "mapping" and isinstance(items, list):
        return {_decode(k, tensors): _decode(v, tensors) for k, v in items}
    if tag == "tuple" and isinstance(items, list):
        return tuple(_decode(item, tensors) for item in items)
    if tag == "list" and isinstance(items, list):
        return [_decode(item, tensors) for item in items]
    raise ValueError("Unknown snapshot state tag.")


def _canonical(value: object) -> bytes:
    """Use a stable JSON representation, rejecting non-finite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(content: bytes) -> str:
    """Hash exact file bytes or the documented canonical JSON payload."""
    return hashlib.sha256(content).hexdigest()


def _file_digest(path: Path) -> str:
    """Hash tensor files without a second whole-file allocation."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_directory(folder: Path) -> None:
    """Persist directory entries before acknowledging a published snapshot."""
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
