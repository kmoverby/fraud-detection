from __future__ import annotations

from typing import Dict


_HIGH_RISK_MERCHANTS = {"gift_cards", "crypto"}


def score_transaction(tx: Dict) -> int:
    """Return a simple fraud risk score from 0 to 100."""
    score = 0

    if tx["device_risk_score"] >= 70:
        score += 25
    elif tx["device_risk_score"] >= 40:
        score += 10

    if tx["is_international"] == 1:
        score += 15

    if tx["amount_usd"] >= 1000:
        score += 25
    elif tx["amount_usd"] >= 500:
        score += 10

    if tx["velocity_24h"] >= 6:
        score += 20
    elif tx["velocity_24h"] >= 3:
        score += 5

    # Prior login failures can signal account takeover.
    if tx["failed_logins_24h"] >= 5:
        score += 20
    elif tx["failed_logins_24h"] >= 2:
        score += 10

    if tx["prior_chargebacks"] >= 2:
        score += 20
    elif tx["prior_chargebacks"] == 1:
        score += 5

    # Very new accounts carry more fraud risk than established ones.
    age = tx.get("account_age_days", 365)
    if age < 30:
        score += 20
    elif age < 90:
        score += 10

    if tx.get("kyc_level") == "basic":
        score += 10

    if tx.get("merchant_category") in _HIGH_RISK_MERCHANTS:
        score += 15

    return max(0, min(score, 100))


def label_risk(score: int) -> str:
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"
