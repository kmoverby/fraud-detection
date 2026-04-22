import pytest

from risk_rules import label_risk, score_transaction


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_tx(**overrides):
    """Fully benign transaction that scores 0. Override one field at a time to
    isolate each signal's contribution."""
    base = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
        "account_age_days": 365,
        "kyc_level": "full",
        "merchant_category": "grocery",
    }
    base.update(overrides)
    return base


# ── label_risk ────────────────────────────────────────────────────────────────

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# ── Score bounds ──────────────────────────────────────────────────────────────

def test_all_clean_signals_score_zero():
    assert score_transaction(_clean_tx()) == 0


def test_all_risky_signals_score_high():
    tx = {
        "device_risk_score": 80,
        "is_international": 1,
        "amount_usd": 1500,
        "velocity_24h": 8,
        "failed_logins_24h": 6,
        "prior_chargebacks": 3,
        "account_age_days": 10,
        "kyc_level": "basic",
        "merchant_category": "gift_cards",
    }
    assert score_transaction(tx) == 100


def test_score_never_exceeds_100():
    tx = {
        "device_risk_score": 99,
        "is_international": 1,
        "amount_usd": 9999,
        "velocity_24h": 99,
        "failed_logins_24h": 99,
        "prior_chargebacks": 99,
        "account_age_days": 1,
        "kyc_level": "basic",
        "merchant_category": "crypto",
    }
    assert score_transaction(tx) <= 100


def test_score_never_below_zero():
    assert score_transaction(_clean_tx(device_risk_score=0, amount_usd=0)) >= 0


# ── Exact signal contributions ────────────────────────────────────────────────
# Each test activates exactly one signal on an otherwise clean transaction so
# the score equals the signal's documented point value.  This catches silent
# weight changes — e.g. someone changing += 25 to += 35.

@pytest.mark.parametrize("field,value,expected_pts", [
    # device risk
    ("device_risk_score", 75,  25),
    ("device_risk_score", 50,  10),
    # international origin
    ("is_international",  1,   15),
    # purchase amount
    ("amount_usd",        1200, 25),
    ("amount_usd",        600,  10),
    # transaction velocity
    ("velocity_24h",      7,   20),
    ("velocity_24h",      4,    5),
    # failed logins
    ("failed_logins_24h", 6,   20),
    ("failed_logins_24h", 3,   10),
    # prior chargeback history
    ("prior_chargebacks", 3,   20),
    ("prior_chargebacks", 1,    5),
    # account age
    ("account_age_days",  15,  20),
    ("account_age_days",  60,  10),
    # KYC level
    ("kyc_level",         "basic", 10),
    # high-risk merchant category
    ("merchant_category", "gift_cards", 15),
    ("merchant_category", "crypto",     15),
])
def test_signal_exact_contribution(field, value, expected_pts):
    score = score_transaction(_clean_tx(**{field: value}))
    assert score == expected_pts, (
        f"{field}={value!r}: expected {expected_pts} pts, got {score}"
    )


# ── Tier boundary conditions ──────────────────────────────────────────────────
# Tests the value immediately below each threshold (should NOT trigger the
# tier) and the value AT the threshold (MUST trigger the tier).
# This catches off-by-one errors such as > vs >=.

@pytest.mark.parametrize("field,just_below,at_threshold,pts_below,pts_at", [
    ("device_risk_score",  39,  40,  0,   10),   # medium device tier at 40
    ("device_risk_score",  69,  70,  10,  25),   # high device tier at 70
    ("amount_usd",        499, 500,  0,   10),   # medium amount tier at 500
    ("amount_usd",        999, 1000, 10,  25),   # high amount tier at 1000
    ("velocity_24h",        2,   3,  0,    5),   # medium velocity tier at 3
    ("velocity_24h",        5,   6,  5,   20),   # high velocity tier at 6
    ("failed_logins_24h",   1,   2,  0,   10),   # medium login tier at 2
    ("failed_logins_24h",   4,   5,  10,  20),   # high login tier at 5
    ("prior_chargebacks",   0,   1,  0,    5),   # first chargeback at 1
    ("prior_chargebacks",   1,   2,  5,   20),   # repeat-offender tier at 2
])
def test_tier_boundary(field, just_below, at_threshold, pts_below, pts_at):
    score_below = score_transaction(_clean_tx(**{field: just_below}))
    score_at    = score_transaction(_clean_tx(**{field: at_threshold}))
    assert score_below == pts_below, (
        f"{field}={just_below} (below threshold): expected {pts_below} pts, got {score_below}"
    )
    assert score_at == pts_at, (
        f"{field}={at_threshold} (at threshold): expected {pts_at} pts, got {score_at}"
    )


def test_account_age_boundary_at_30():
    """age=29 triggers the < 30 (very new) tier; age=30 drops to the < 90 tier."""
    assert score_transaction(_clean_tx(account_age_days=29)) == 20
    assert score_transaction(_clean_tx(account_age_days=30)) == 10


def test_account_age_boundary_at_90():
    """age=89 still triggers the < 90 tier; age=90 adds nothing."""
    assert score_transaction(_clean_tx(account_age_days=89)) == 10
    assert score_transaction(_clean_tx(account_age_days=90)) == 0


# ── Monotonicity ──────────────────────────────────────────────────────────────
# For every continuous risk signal, increasing the value must never LOWER the
# score.  This property is fundamental to a fraud scorer — riskier inputs must
# not produce lower or equal scores than safer inputs at a lower tier.

def test_device_risk_score_monotone():
    values = [0, 10, 39, 40, 69, 70, 90, 100]
    scores = [score_transaction(_clean_tx(device_risk_score=v)) for v in values]
    assert scores == sorted(scores), \
        f"device_risk_score not monotone: {list(zip(values, scores))}"


def test_amount_usd_monotone():
    values = [0, 100, 499, 500, 999, 1000, 5000]
    scores = [score_transaction(_clean_tx(amount_usd=v)) for v in values]
    assert scores == sorted(scores), \
        f"amount_usd not monotone: {list(zip(values, scores))}"


def test_velocity_24h_monotone():
    values = [0, 1, 2, 3, 5, 6, 15]
    scores = [score_transaction(_clean_tx(velocity_24h=v)) for v in values]
    assert scores == sorted(scores), \
        f"velocity_24h not monotone: {list(zip(values, scores))}"


def test_failed_logins_24h_monotone():
    values = [0, 1, 2, 4, 5, 10]
    scores = [score_transaction(_clean_tx(failed_logins_24h=v)) for v in values]
    assert scores == sorted(scores), \
        f"failed_logins_24h not monotone: {list(zip(values, scores))}"


def test_prior_chargebacks_monotone():
    values = [0, 1, 2, 5, 10]
    scores = [score_transaction(_clean_tx(prior_chargebacks=v)) for v in values]
    assert scores == sorted(scores), \
        f"prior_chargebacks not monotone: {list(zip(values, scores))}"


def test_account_age_inverse_monotone():
    # Younger accounts are riskier, so score must be non-increasing as age grows.
    values = [1, 15, 29, 30, 60, 89, 90, 365]
    scores = [score_transaction(_clean_tx(account_age_days=v)) for v in values]
    assert scores == sorted(scores, reverse=True), \
        f"account_age_days not inverse-monotone: {list(zip(values, scores))}"


# ── Optional field defaults ───────────────────────────────────────────────────
# account_age_days, kyc_level, and merchant_category use .get() with safe
# defaults.  If a caller omits them the function must not raise and must
# produce the same score as an established, full-KYC, low-risk merchant.

def test_score_transaction_without_optional_fields():
    tx_without = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    assert score_transaction(tx_without) == 0


def test_missing_account_age_defaults_to_established():
    tx_without = {
        "device_risk_score": 10, "is_international": 0, "amount_usd": 50,
        "velocity_24h": 1, "failed_logins_24h": 0, "prior_chargebacks": 0,
    }
    tx_with_365 = _clean_tx(account_age_days=365)
    assert score_transaction(tx_without) == score_transaction(tx_with_365)


def test_missing_kyc_level_defaults_to_full():
    tx_without = {
        "device_risk_score": 10, "is_international": 0, "amount_usd": 50,
        "velocity_24h": 1, "failed_logins_24h": 0, "prior_chargebacks": 0,
        "account_age_days": 365, "merchant_category": "grocery",
    }
    tx_with_full = _clean_tx(kyc_level="full")
    assert score_transaction(tx_without) == score_transaction(tx_with_full)


def test_missing_merchant_category_defaults_to_safe():
    tx_without = {
        "device_risk_score": 10, "is_international": 0, "amount_usd": 50,
        "velocity_24h": 1, "failed_logins_24h": 0, "prior_chargebacks": 0,
        "account_age_days": 365, "kyc_level": "full",
    }
    tx_with_grocery = _clean_tx(merchant_category="grocery")
    assert score_transaction(tx_without) == score_transaction(tx_with_grocery)


# ── Failed logins signal ──────────────────────────────────────────────────────
# failed_logins_24h has no dedicated direction or coverage tests elsewhere.

def test_failed_logins_below_threshold_adds_no_risk():
    assert score_transaction(_clean_tx(failed_logins_24h=0)) == 0
    assert score_transaction(_clean_tx(failed_logins_24h=1)) == 0


def test_failed_logins_raises_score():
    no_logins  = score_transaction(_clean_tx(failed_logins_24h=0))
    mid_logins = score_transaction(_clean_tx(failed_logins_24h=3))
    many_logins = score_transaction(_clean_tx(failed_logins_24h=6))
    assert mid_logins > no_logins,  "2+ failed logins should raise the score"
    assert many_logins > mid_logins, "5+ failed logins should raise the score above the 2+ tier"


def test_failed_logins_raises_more_than_medium_tier():
    mid   = score_transaction(_clean_tx(failed_logins_24h=3))
    high_ = score_transaction(_clean_tx(failed_logins_24h=6))
    assert high_ > mid


# ── Individual signal direction checks ───────────────────────────────────────
# Regression guards that each signal moves the score in the correct direction.

def test_high_device_risk_raises_score():
    assert score_transaction(_clean_tx(device_risk_score=75)) > \
           score_transaction(_clean_tx(device_risk_score=10))


def test_high_device_risk_raises_more_than_medium():
    assert score_transaction(_clean_tx(device_risk_score=75)) > \
           score_transaction(_clean_tx(device_risk_score=50))


def test_international_transaction_raises_score():
    assert score_transaction(_clean_tx(is_international=1)) > \
           score_transaction(_clean_tx(is_international=0))


def test_large_amount_adds_risk():
    assert score_transaction(_clean_tx(amount_usd=1200)) >= 25


def test_medium_amount_adds_moderate_risk():
    score_600  = score_transaction(_clean_tx(amount_usd=600))
    score_1200 = score_transaction(_clean_tx(amount_usd=1200))
    assert score_600 >= 10
    assert score_600 < score_1200


def test_high_velocity_raises_score():
    assert score_transaction(_clean_tx(velocity_24h=7)) > \
           score_transaction(_clean_tx(velocity_24h=1))


def test_very_high_velocity_raises_more_than_medium_velocity():
    assert score_transaction(_clean_tx(velocity_24h=7)) > \
           score_transaction(_clean_tx(velocity_24h=4))


def test_prior_chargebacks_raise_score():
    no_cb   = score_transaction(_clean_tx(prior_chargebacks=0))
    one_cb  = score_transaction(_clean_tx(prior_chargebacks=1))
    many_cb = score_transaction(_clean_tx(prior_chargebacks=3))
    assert one_cb  > no_cb
    assert many_cb > one_cb


def test_very_new_account_raises_score():
    assert score_transaction(_clean_tx(account_age_days=15)) > \
           score_transaction(_clean_tx(account_age_days=365))


def test_new_account_raises_more_than_medium_age():
    assert score_transaction(_clean_tx(account_age_days=15)) > \
           score_transaction(_clean_tx(account_age_days=60))


def test_established_account_adds_no_risk():
    assert score_transaction(_clean_tx(account_age_days=90)) == \
           score_transaction(_clean_tx(account_age_days=500))


def test_basic_kyc_raises_score():
    assert score_transaction(_clean_tx(kyc_level="basic")) > \
           score_transaction(_clean_tx(kyc_level="full"))


def test_gift_cards_merchant_raises_score():
    assert score_transaction(_clean_tx(merchant_category="gift_cards")) > \
           score_transaction(_clean_tx(merchant_category="grocery"))


def test_crypto_merchant_raises_score():
    assert score_transaction(_clean_tx(merchant_category="crypto")) > \
           score_transaction(_clean_tx(merchant_category="grocery"))


def test_standard_merchant_adds_no_risk():
    assert score_transaction(_clean_tx(merchant_category="grocery")) == \
           score_transaction(_clean_tx(merchant_category="electronics")) == \
           score_transaction(_clean_tx(merchant_category="travel"))
