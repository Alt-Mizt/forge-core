"""Build the environment for one provider call, holding one account's credentials only.

A worker process starts from a small fixed environment plus the credentials of the one
account it acts for. forge stores where credentials are, never the secrets themselves.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from forge.accounts import Account

BASICS = ("PATH", "HOME", "LANG", "PYTHONPATH", "SSL_CERT_FILE", "SSL_CERT_DIR")
MODAL_TOKENS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
# Permission bits for group and other users; a token file must have none of them set.
SHARED_PERMISSIONS = 0o077


def environment(account: Account) -> dict[str, str]:
    """The basic environment plus this account's credentials."""
    basics = {name: os.environ[name] for name in BASICS if name in os.environ}
    if account.provider == "kaggle":
        return basics | _kaggle(account)
    if account.provider == "modal":
        return basics | _modal(account)
    raise ValueError(f"{account.provider} accounts need no credentials.")


def _kaggle(account: Account) -> dict[str, str]:
    """Kaggle's token, read from the account's token file at the moment of the call.

    An empty config folder keeps the SDK from falling back to any other Kaggle login.
    """
    if account.credentials is None:
        raise ValueError(f"Account {account.name} has no Kaggle token file.")
    path = Path(account.credentials).expanduser()
    if not path.is_file():
        raise ValueError(f"Account {account.name}'s Kaggle token file is missing.")
    if path.stat().st_mode & SHARED_PERMISSIONS:
        raise ValueError(f"Make {path.name} private first: chmod 600 {path.name}")
    values = {"KAGGLE_CONFIG_DIR": tempfile.mkdtemp(prefix="forge-kaggle-")}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        values |= {"KAGGLE_USERNAME": document["username"], "KAGGLE_KEY": document["key"]}
    else:
        values["KAGGLE_API_TOKEN"] = path.read_text().strip()
    return values


def _modal(account: Account) -> dict[str, str]:
    """Modal tokens from the environment, or the account's profile in Modal's own config."""
    tokens = {name: os.environ[name] for name in MODAL_TOKENS if os.environ.get(name)}
    if len(tokens) == len(MODAL_TOKENS):
        return tokens
    return {"MODAL_PROFILE": account.credentials} if account.credentials else {}
