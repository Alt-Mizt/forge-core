"""Run each provider call in a short-lived worker process and return only its result.

The worker holds one account's credentials. When a call fails, the SDK's own output comes
back with those credentials blanked out, so the message helps without leaking a token.
"""

import json
import shutil
import subprocess
import sys
from typing import Any

from forge.accounts import Account
from forge.providers.credentials import environment

DEFAULT_TIMEOUT_SECONDS = 45
SECRET_NAMES = ("KAGGLE_KEY", "KAGGLE_API_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
ERROR_LINES = 8


def call(
    account: Account,
    operation: str,
    request: dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one operation for one account and return its result object.

    Raises:
        ValueError: When the call fails, with the end of the worker's output.
    """
    environment_ = environment(account)
    try:
        response = subprocess.run(
            [sys.executable, "-m", "forge.providers.worker", account.provider, operation],
            input=json.dumps(request, allow_nan=False),
            capture_output=True,
            text=True,
            env=environment_,
            timeout=timeout,
            check=False,
        )
    finally:
        if "KAGGLE_CONFIG_DIR" in environment_:
            shutil.rmtree(environment_["KAGGLE_CONFIG_DIR"], ignore_errors=True)
    if response.returncode:
        detail = _redacted(response.stderr, environment_)
        raise ValueError(
            f"{account.name}: the {account.provider} {operation} call failed.\n{detail}"
        )
    try:
        result = json.loads(response.stdout)
    except json.JSONDecodeError:
        raise ValueError(
            f"{account.name}: the {account.provider} worker returned no result."
        ) from None
    if not isinstance(result, dict):
        raise ValueError(
            f"{account.name}: the {account.provider} worker returned an invalid result."
        )
    return result


def _redacted(text: str, environment_: dict[str, str]) -> str:
    """The last lines of the worker's output, with the account's secrets blanked out."""
    for name in SECRET_NAMES:
        if secret := environment_.get(name):
            text = text.replace(secret, "***")
    lines = text.strip().splitlines()[-ERROR_LINES:]
    return "".join(f"  {line}\n" for line in lines)
