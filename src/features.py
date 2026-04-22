from __future__ import annotations

import pandas as pd


def build_model_frame(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(accounts, on="account_id", how="left")

    unmatched = df["account_id"][df["prior_chargebacks"].isna()]
    if not unmatched.empty:
        raise ValueError(f"Transactions reference unknown account_id(s): {sorted(unmatched.unique())}")

    return df
