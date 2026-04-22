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

def test_all_risky_signals_score_high():
    tx = {
        "device_risk_score": 80,
        "is_international": 1,
        "amount_usd": 1500,
        "velocity_24h": 8,
        "failed_logins_24h": 6,
        "prior_chargebacks": 3,
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
    }
    assert score_transaction(tx) <= 100


def test_score_never_below_zero():
    tx = _clean_tx(device_risk_score=0, amount_usd=0)
    assert score_transaction(tx) >= 0
