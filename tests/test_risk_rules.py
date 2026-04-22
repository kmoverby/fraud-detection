from risk_rules import label_risk, score_transaction


# --- label_risk ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# --- baseline helpers ---

def _clean_tx(**overrides):
    """A fully benign transaction; override individual fields to test one signal at a time."""
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


# --- amount (known-good signal, regression guard) ---

def test_large_amount_adds_risk():
    tx = _clean_tx(amount_usd=1200)
    assert score_transaction(tx) >= 25


def test_medium_amount_adds_moderate_risk():
    tx = _clean_tx(amount_usd=600)
    score = score_transaction(tx)
    assert score >= 10
    assert score < score_transaction(_clean_tx(amount_usd=1200))


# --- device risk (Bug 1) ---

def test_high_device_risk_raises_score():
    low_device = score_transaction(_clean_tx(device_risk_score=10))
    high_device = score_transaction(_clean_tx(device_risk_score=75))
    assert high_device > low_device, "high device risk should raise the score, not lower it"


def test_high_device_risk_raises_more_than_medium():
    medium_device = score_transaction(_clean_tx(device_risk_score=50))
    high_device = score_transaction(_clean_tx(device_risk_score=75))
    assert high_device > medium_device


# --- international transactions (Bug 2) ---

def test_international_transaction_raises_score():
    domestic = score_transaction(_clean_tx(is_international=0))
    international = score_transaction(_clean_tx(is_international=1))
    assert international > domestic, "international transactions should raise the score, not lower it"


# --- velocity (Bug 3) ---

def test_high_velocity_raises_score():
    low_vel = score_transaction(_clean_tx(velocity_24h=1))
    high_vel = score_transaction(_clean_tx(velocity_24h=7))
    assert high_vel > low_vel, "high velocity should raise the score, not lower it"


def test_very_high_velocity_raises_more_than_medium_velocity():
    medium_vel = score_transaction(_clean_tx(velocity_24h=4))
    high_vel = score_transaction(_clean_tx(velocity_24h=7))
    assert high_vel > medium_vel


# --- prior chargebacks (Bug 4) ---

def test_prior_chargebacks_raise_score():
    no_cb = score_transaction(_clean_tx(prior_chargebacks=0))
    one_cb = score_transaction(_clean_tx(prior_chargebacks=1))
    many_cb = score_transaction(_clean_tx(prior_chargebacks=3))
    assert one_cb > no_cb, "one prior chargeback should raise the score, not lower it"
    assert many_cb > one_cb, "more prior chargebacks should raise the score higher"


# --- compound case ---

# --- account age (new signal) ---

def test_very_new_account_raises_score():
    established = score_transaction(_clean_tx(account_age_days=365))
    very_new = score_transaction(_clean_tx(account_age_days=15))
    assert very_new > established, "account < 30 days old should raise the score"


def test_new_account_raises_more_than_medium_age():
    medium_age = score_transaction(_clean_tx(account_age_days=60))
    very_new = score_transaction(_clean_tx(account_age_days=15))
    assert very_new > medium_age


def test_established_account_adds_no_risk():
    score_90 = score_transaction(_clean_tx(account_age_days=90))
    score_500 = score_transaction(_clean_tx(account_age_days=500))
    assert score_90 == score_500, "accounts >= 90 days old should not differ in score"


# --- kyc_level (new signal) ---

def test_basic_kyc_raises_score():
    full_kyc = score_transaction(_clean_tx(kyc_level="full"))
    basic_kyc = score_transaction(_clean_tx(kyc_level="basic"))
    assert basic_kyc > full_kyc, "basic KYC should raise the score vs full KYC"


# --- merchant_category (new signal) ---

def test_gift_cards_merchant_raises_score():
    low_risk = score_transaction(_clean_tx(merchant_category="grocery"))
    high_risk = score_transaction(_clean_tx(merchant_category="gift_cards"))
    assert high_risk > low_risk, "gift_cards merchant should raise the score"


def test_crypto_merchant_raises_score():
    low_risk = score_transaction(_clean_tx(merchant_category="grocery"))
    high_risk = score_transaction(_clean_tx(merchant_category="crypto"))
    assert high_risk > low_risk, "crypto merchant should raise the score"


def test_standard_merchant_adds_no_risk():
    grocery = score_transaction(_clean_tx(merchant_category="grocery"))
    electronics = score_transaction(_clean_tx(merchant_category="electronics"))
    travel = score_transaction(_clean_tx(merchant_category="travel"))
    assert grocery == electronics == travel


# --- compound case ---

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
    assert score_transaction(tx) == 100, "all risky signals should hit the 100 ceiling"


def test_all_clean_signals_score_zero():
    tx = _clean_tx()
    assert score_transaction(tx) == 0, "a fully benign transaction should score 0"


# --- score is always within bounds ---

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
    tx = _clean_tx(device_risk_score=0, amount_usd=0)
    assert score_transaction(tx) >= 0
