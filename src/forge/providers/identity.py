"""Confirm that Kaggle ran the private notebook forge asked for, and no other."""

from typing import Any

NAME_PARTS = 2


def confirm_notebook(expected: str, receipt: str, metadata: dict[str, Any]) -> None:
    """Accept a Kaggle submission only if it names the expected notebook and is private.

    ``expected`` is ``owner/name``. Kaggle's receipt is a path such as ``/code/owner/name``,
    and it can be empty; Kaggle's own record of the notebook then has to match on its own.
    """
    if receipt and _name(receipt) != expected:
        raise ValueError("The submission receipt names a different notebook.")
    if metadata.get("ref") != expected or metadata.get("private") is not True:
        raise ValueError("The provider did not confirm this notebook's private canonical identity.")


def _name(receipt: str) -> str:
    """The ``owner/name`` at the end of a receipt path."""
    return "/".join(receipt.strip("/").split("/")[-NAME_PARTS:])
