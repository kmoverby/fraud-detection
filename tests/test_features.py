import pandas as pd
import pytest

from features import build_model_frame


def _accounts(*rows):
    defaults = {"prior_chargebacks": 0, "kyc_level": "full", "account_age_days": 365}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _transactions(*rows):
    defaults = {"amount_usd": 50.0, "failed_logins_24h": 0}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_build_model_frame_merges_account_fields():
    txns = _transactions(
        {"transaction_id": 1, "account_id": 10},
        {"transaction_id": 2, "account_id": 20},
    )
    accts = _accounts(
        {"account_id": 10, "prior_chargebacks": 0, "kyc_level": "full", "account_age_days": 500},
        {"account_id": 20, "prior_chargebacks": 2, "kyc_level": "basic", "account_age_days": 15},
    )
    df = build_model_frame(txns, accts)
    assert len(df) == 2
    assert df.loc[df["account_id"] == 20, "prior_chargebacks"].iloc[0] == 2
    assert df.loc[df["account_id"] == 20, "kyc_level"].iloc[0] == "basic"
    assert df.loc[df["account_id"] == 20, "account_age_days"].iloc[0] == 15


def test_build_model_frame_raises_on_unknown_account_id():
    txns = _transactions(
        {"transaction_id": 1, "account_id": 10},
        {"transaction_id": 2, "account_id": 99},  # 99 has no matching account
    )
    accts = _accounts({"account_id": 10})
    with pytest.raises(ValueError, match="unknown account_id"):
        build_model_frame(txns, accts)


def test_build_model_frame_all_transactions_returned():
    txns = _transactions(
        {"transaction_id": 1, "account_id": 10},
        {"transaction_id": 2, "account_id": 10},
        {"transaction_id": 3, "account_id": 10},
    )
    accts = _accounts({"account_id": 10})
    df = build_model_frame(txns, accts)
    assert len(df) == 3


def test_build_model_frame_no_dead_columns():
    txns = _transactions({"transaction_id": 1, "account_id": 10})
    accts = _accounts({"account_id": 10})
    df = build_model_frame(txns, accts)
    assert "login_pressure" not in df.columns
    assert "is_large_amount" not in df.columns


def test_build_model_frame_one_account_many_transactions():
    """A single account with multiple transactions must produce one output row
    per transaction, with the same account fields repeated on every row."""
    txns = _transactions(
        {"transaction_id": 1, "account_id": 10, "amount_usd": 100.0},
        {"transaction_id": 2, "account_id": 10, "amount_usd": 200.0},
        {"transaction_id": 3, "account_id": 10, "amount_usd": 300.0},
    )
    accts = _accounts({"account_id": 10, "prior_chargebacks": 2, "kyc_level": "basic"})
    df = build_model_frame(txns, accts)
    assert len(df) == 3
    assert (df["prior_chargebacks"] == 2).all(), \
        "account field must be present on every transaction row"
    assert (df["kyc_level"] == "basic").all()
    assert list(df["transaction_id"]) == [1, 2, 3]


def test_build_model_frame_preserves_transaction_fields():
    """Merge must not clobber or drop any transaction-level column."""
    txns = _transactions(
        {"transaction_id": 42, "account_id": 10, "amount_usd": 99.0,
         "failed_logins_24h": 3},
    )
    accts = _accounts({"account_id": 10})
    df = build_model_frame(txns, accts)
    assert df.iloc[0]["transaction_id"] == 42
    assert df.iloc[0]["amount_usd"] == 99.0
    assert df.iloc[0]["failed_logins_24h"] == 3
