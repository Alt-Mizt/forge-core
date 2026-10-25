"""Accounts keep provider defaults, references to credentials, and nothing secret."""

import json
from decimal import Decimal

import pytest

from forge import accounts
from forge.providers import credentials


def test_accounts_round_trip_through_the_config_file(tmp_path):
    path = tmp_path / "accounts.toml"
    saved = [
        accounts.new_account("kaggle", credentials="/token/file"),
        accounts.new_account("local", device="cpu", session_limit=10),
        accounts.new_account("modal", paid_cap=Decimal("5")),
    ]
    accounts.save(path, saved)
    assert accounts.load(path) == saved


def test_provider_defaults_fill_in_what_was_not_given():
    kaggle = accounts.new_account("kaggle")
    assert (kaggle.name, kaggle.allowance, kaggle.paid_cap) == ("kaggle", Decimal(30), 0)
    local = accounts.new_account("local")
    assert local.allowance is None and local.limit() is None


def test_an_accounts_limit_is_its_free_time_plus_its_paid_cap():
    limit = accounts.new_account("modal", paid_cap=Decimal(5)).limit()
    assert limit is not None and limit.amount == Decimal(35)


@pytest.mark.parametrize("name", ["Kaggle", "a b", "9lives", ""])
def test_unusable_names_refuse(name):
    with pytest.raises(ValueError, match="names"):
        accounts.new_account("kaggle", name=name or "-")


def test_unknown_providers_and_negative_caps_refuse():
    with pytest.raises(ValueError, match="Unknown provider"):
        accounts.new_account("cloud")
    with pytest.raises(ValueError, match="negative"):
        accounts.new_account("modal", paid_cap=Decimal(-1))


def test_a_kaggle_call_carries_only_that_accounts_token(tmp_path, monkeypatch):
    token = tmp_path / "kaggle.json"
    token.write_text(json.dumps({"username": "someone", "key": "example-key"}))
    token.chmod(0o600)
    monkeypatch.setenv("MODAL_TOKEN_ID", "unrelated")
    account = accounts.new_account("kaggle", credentials=str(token))
    environment = credentials.environment(account)
    assert environment["KAGGLE_USERNAME"] == "someone"
    assert "MODAL_TOKEN_ID" not in environment


def test_a_token_file_others_can_read_is_refused(tmp_path):
    token = tmp_path / "access_token"
    token.write_text("example")
    token.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        credentials.environment(accounts.new_account("kaggle", credentials=str(token)))


def test_a_modal_call_names_the_profile_without_reading_it(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    account = accounts.new_account("modal", credentials="work")
    assert credentials.environment(account)["MODAL_PROFILE"] == "work"
