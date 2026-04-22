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
