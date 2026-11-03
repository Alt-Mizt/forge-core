"""Kaggle SDK calls, run only inside a short-lived worker process (see ``bridge``).

Every notebook is private, runs on Kaggle's free T4 machine and carries Kaggle's own
session timeout. Paid scaling refuses. The worker returns selected fields only.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest,
    ApiGetKernelRequest,
    ApiSaveKernelRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

from forge.providers.identity import confirm_notebook

ACCELERATOR = "NvidiaTeslaT4"
RUNNING_STATUSES = frozenset({"QUEUED", "RUNNING", "NEW_SCRIPT"})
OWNER = re.compile(r"[A-Za-z0-9_-]{1,64}")
OUT_FOLDER = "forge/out"
SECONDS_PER_HOUR = 3600


class Client:
    """One Kaggle account, authenticated from this worker's environment."""

    def __init__(self) -> None:
        """Authenticate with the token in this worker's environment."""
        self.api = KaggleApi()
        self.api.authenticate()

    def invoke(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run one operation: check, start, poll, fetch or stop."""
        operations = {
            "check": self._check,
            "start": self._start,
            "poll": self._poll,
            "fetch": self._fetch,
            "stop": self._stop,
        }
        if operation not in operations:
            raise ValueError("Unknown Kaggle operation.")
        return operations[operation](request)

    def _check(self, request: dict[str, Any]) -> dict[str, Any]:
        """Read the free GPU quota, which also proves the credentials work."""
        del request
        quota = self._quota()
        hours = quota["free_seconds"] / SECONDS_PER_HOUR
        return {"note": f"{hours:.1f} GPU-hours of Kaggle quota left this week"}

    def _start(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit a private notebook, if the free quota covers its whole time limit."""
        if self._quota()["free_seconds"] < request["seconds"]:
            raise ValueError("Kaggle's free quota cannot cover this session's time limit.")
        ref = self._ref(request["handle"])
        with self.api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(_notebook(ref, request))
            if response.error:
                raise ValueError("Kaggle did not accept the notebook.")
        metadata = self._metadata(ref)
        confirm_notebook(ref, response.ref, metadata)
        if metadata["machine_shape"] != ACCELERATOR:
            raise ValueError("Kaggle did not confirm the requested GPU machine.")
        return {"ref": ref}

    def _poll(self, request: dict[str, Any]) -> dict[str, Any]:
        """Report running, failed, or how the job ended, read from its exit code."""
        ref = self._ref(request["handle"])
        status = self.api.kernels_status(ref).status.name
        if status in RUNNING_STATUSES:
            return {"state": "running"}
        if status != "COMPLETE":
            return {"state": "failed"}
        with tempfile.TemporaryDirectory() as folder:
            self.api.kernels_output(
                ref, folder, file_pattern=f"^{OUT_FOLDER}/exit_code$", quiet=True
            )
            code = Path(folder, OUT_FOLDER, "exit_code")
            return {"exit_code": int(code.read_text())} if code.exists() else {"state": "failed"}

    def _fetch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Download the session's ``out`` folder into ``request["directory"]``."""
        ref = self._ref(request["handle"])
        destination = Path(request["directory"])
        with tempfile.TemporaryDirectory() as folder:
            self.api.kernels_output(ref, folder, file_pattern=f"^{OUT_FOLDER}/", quiet=True)
            source = Path(folder, OUT_FOLDER)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                destination.mkdir(parents=True)
        return {"fetched": True}

    def _stop(self, request: dict[str, Any]) -> dict[str, Any]:
        """Confirm the notebook has ended and holds no GPU time.

        Kaggle offers no way to cancel a notebook from here, so a running one is reported
        as not stopped; its own session timeout ends it.
        """
        ref = self._ref(request["handle"])
        self._require_private(ref)
        status = self.api.kernels_status(ref).status.name
        ended = status not in RUNNING_STATUSES and self._quota()["reserved_seconds"] == 0
        return {"stopped": ended}

    def _ref(self, handle: str) -> str:
        """The notebook's full name: the account's user name, then the session handle."""
        owner = self.api.config_values["username"]
        if OWNER.fullmatch(owner) is None:
            raise ValueError("The Kaggle account has no valid user name.")
        return f"{owner}/{handle}"

    def _metadata(self, ref: str) -> dict[str, Any]:
        """Kaggle's record of a notebook: its name, privacy and machine; unknown refuses."""
        with self.api.build_kaggle_client() as client:
            lookup = ApiGetKernelRequest()
            lookup.user_name, lookup.kernel_slug = ref.split("/")
            metadata = client.kernels.kernels_api_client.get_kernel(lookup).metadata
        if metadata is None:
            raise ValueError("Kaggle does not know this notebook.")
        return {
            "ref": metadata.ref,
            "private": metadata.is_private,
            "machine_shape": metadata.machine_shape,
        }

    def _require_private(self, ref: str) -> None:
        """Refuse unless Kaggle confirms the notebook exists under this name and is private."""
        confirm_notebook(ref, "", self._metadata(ref))

    def _quota(self) -> dict[str, Any]:
        """Return the free GPU seconds left and reserved, refusing if paid scaling is on."""
        with self.api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest()
            )
        quota = response.gpu_quota
        if quota is None or quota.is_pay_to_scale_enabled:
            raise ValueError("Kaggle's free GPU quota is unknown or paid scaling is on.")
        durations = (quota.total_time_allowed, quota.time_used, quota.time_reserved)
        if any(duration is None for duration in durations):
            raise ValueError("Kaggle has not confirmed the free GPU quota left.")
        left = durations[0] - durations[1] - durations[2]
        return {
            "free_seconds": left.total_seconds(),
            "reserved_seconds": quota.time_reserved.total_seconds(),
        }


def _notebook(ref: str, request: dict[str, Any]) -> ApiSaveKernelRequest:
    """Describe a private GPU script that runs once, with Kaggle's own session timeout."""
    notebook = ApiSaveKernelRequest()
    notebook.slug = ref
    notebook.new_title = ref.split("/")[1]
    notebook.text = request["code"]
    notebook.language = "python"
    notebook.kernel_type = "script"
    notebook.is_private = True
    notebook.enable_gpu = True
    # The notebook installs torch with pip, so it needs the internet.
    notebook.enable_internet = True
    notebook.machine_shape = ACCELERATOR
    notebook.session_timeout_seconds = request["seconds"]
    notebook.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    return notebook
