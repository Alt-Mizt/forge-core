"""Modal SDK calls, run only inside a short-lived worker process (see ``bridge``).

A session is one sandbox: the GPU forge chose, capped CPU and memory, no network, gVisor,
and a timeout that Modal enforces. The image has the session's torch version baked in. The
sandbox waits for its folder, runs the session, then stays up a short while so forge can
collect the results.
"""

from __future__ import annotations

import base64
from typing import Any

import modal

from forge.accounts import MODAL_SANDBOX_CORES, MODAL_SANDBOX_GIB

APP_NAME = "forge"
MIB_PER_GIB = 1024
# The sandbox asks for Modal's smallest default and may grow up to forge's cap, which is
# what bounds a session's price (see forge.accounts).
CPU = (0.125, float(MODAL_SANDBOX_CORES))
MEMORY_MIB = (128, MODAL_SANDBOX_GIB * MIB_PER_GIB)
TAG = "forge_session"
PYTHON = "3.11"
# After the session ends, the sandbox waits this long for forge to collect its results.
COLLECT_SECONDS = 120
# Reading a file the session has not written yet fails with one of these.
MISSING = (FileNotFoundError, modal.exception.Error)
RESULT_FILES = (
    "exit_code",
    "log.txt",
    "outputs.zip",
    "snapshot/snapshot.json",
    "snapshot/tensors.safetensors",
)
COMMAND = (
    "while [ ! -f /forge.ready ]; do sleep 1; done; "
    "python -c \"import zipfile; zipfile.ZipFile('/forge.zip').extractall('/forge')\"; "
    "cd /forge && PYTHONPATH=/forge python -m forge.session /forge {seconds}; "
    "sleep {collect}"
)
PAUSE = "import os, signal; os.kill(int(open('/forge/out/runner.pid').read()), signal.SIGTERM)"


class Client:
    """One Modal workspace, authenticated from this worker's environment."""

    def invoke(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run one operation: check, start, poll, fetch, logs, pause or stop."""
        operations = {
            "check": self._check,
            "start": self._start,
            "poll": self._poll,
            "fetch": self._fetch,
            "logs": self._logs,
            "pause": self._pause,
            "stop": self._stop,
        }
        if operation not in operations:
            raise ValueError("Unknown Modal operation.")
        return operations[operation](request)

    def _check(self, request: dict[str, Any]) -> dict[str, Any]:
        """Look up forge's app; an authenticated 'not found' also proves the token works."""
        del request
        try:
            modal.App.lookup(APP_NAME)
        except modal.exception.NotFoundError:
            return {"note": "authenticated; no forge sessions yet"}
        return {"note": "authenticated"}

    def _start(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create the sandbox with a Modal-enforced timeout, then hand it its folder."""
        seconds = int(request["seconds"])
        packages = [f"torch=={request['torch']}", "numpy", "safetensors", *request["requirements"]]
        image = modal.Image.debian_slim(python_version=PYTHON).pip_install(*packages)
        command = COMMAND.format(seconds=seconds, collect=COLLECT_SECONDS)
        sandbox = modal.Sandbox.create(
            "sh",
            "-c",
            command,
            app=modal.App.lookup(APP_NAME, create_if_missing=True),
            image=image,
            gpu=request["gpu"],
            cpu=CPU,
            memory=MEMORY_MIB,
            runtime="gvisor",
            timeout=seconds + COLLECT_SECONDS,
            block_network=True,
            name=request["handle"],
            tags={TAG: request["handle"]},
        )
        sandbox.filesystem.write_bytes(base64.b64decode(request["bundle"]), "/forge.zip")
        sandbox.filesystem.write_bytes(b"", "/forge.ready")
        return {"id": sandbox.object_id}

    def _poll(self, request: dict[str, Any]) -> dict[str, Any]:
        """Running until the session wrote its exit code; failed if the sandbox is gone."""
        sandbox = _sandbox(request["handle"])
        if sandbox is None or sandbox.poll() is not None:
            return {"state": "failed"}
        try:
            code = sandbox.filesystem.read_text("/forge/out/exit_code")
        except MISSING:
            return {"state": "running"}
        return {"exit_code": int(code)}

    def _fetch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Read the result files, base64-encoded; files the session did not write are absent."""
        sandbox = _sandbox(request["handle"])
        if sandbox is None:
            raise ValueError("The Modal sandbox is gone; its results were not collected in time.")
        files = {}
        for name in RESULT_FILES:
            try:
                content = sandbox.filesystem.read_bytes(f"/forge/out/{name}")
            except MISSING:
                continue
            files[name] = base64.b64encode(content).decode()
        return {"files": files}

    def _logs(self, request: dict[str, Any]) -> dict[str, Any]:
        """The job's log so far."""
        sandbox = _sandbox(request["handle"])
        try:
            text = sandbox.filesystem.read_text("/forge/out/log.txt") if sandbox else ""
        except MISSING:
            text = ""
        return {"text": text}

    def _pause(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send SIGTERM to the wrapped job, which saves after its current batch."""
        sandbox = _sandbox(request["handle"])
        if sandbox is None:
            raise ValueError("The Modal sandbox is gone.")
        sandbox.exec("python", "-c", PAUSE).wait()
        return {"paused": True}

    def _stop(self, request: dict[str, Any]) -> dict[str, Any]:
        """Terminate the session's sandboxes and confirm none is still running."""
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        tags = {TAG: request["handle"]}
        for sandbox in modal.Sandbox.list(app_id=app.app_id, tags=tags):
            sandbox.terminate(wait=True)
        remaining = list(modal.Sandbox.list(app_id=app.app_id, tags=tags))
        return {"stopped": all(sandbox.poll() is not None for sandbox in remaining)}


def _sandbox(handle: str) -> modal.Sandbox | None:
    """The sandbox running a session, found by its tag."""
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandboxes = list(modal.Sandbox.list(app_id=app.app_id, tags={TAG: handle}))
    return sandboxes[0] if sandboxes else None
