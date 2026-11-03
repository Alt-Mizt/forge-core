"""A missing short receipt is recoverable only through the same private notebook."""

import pytest

from forge.providers.identity import confirm_notebook


def test_a_private_canonical_identity_recovers_an_omitted_submission_ref():
    confirm_notebook("owner/run", "", {"ref": "owner/run", "private": True})


@pytest.mark.parametrize(
    "metadata",
    [
        {"ref": "owner/another-run", "private": True},
        {"ref": "owner/run", "private": False},
    ],
)
def test_missing_receipts_cannot_be_confirmed_by_a_different_or_public_notebook(metadata):
    with pytest.raises(ValueError, match="identity"):
        confirm_notebook("owner/run", "", metadata)


@pytest.mark.parametrize("receipt", ["owner/run", "/owner/run", "/code/owner/run"])
def test_a_receipt_path_ending_in_the_notebook_name_is_accepted(receipt):
    confirm_notebook("owner/run", receipt, {"ref": "owner/run", "private": True})


def test_a_receipt_for_another_notebook_is_refused():
    with pytest.raises(ValueError, match="different notebook"):
        confirm_notebook("owner/run", "/code/owner/other", {"ref": "owner/run", "private": True})
