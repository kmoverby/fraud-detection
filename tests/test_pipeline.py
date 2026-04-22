"""
End-to-end integration tests that run the full pipeline on the real CSV data.
These act as a regression safety net: if any future change to scoring logic,
feature engineering, or data loading breaks the business-critical outcomes,
these tests will fail before the code ships.
"""

import pytest

from analyze_fraud import load_inputs, score_transactions, summarize_results


# IDs of transactions that are confirmed chargebacks in chargebacks.csv.
KNOWN_CHARGEBACK_IDS = {50003, 50006, 50008, 50011, 50013, 50014, 50015, 50019}

# IDs of transactions with no chargeback in chargebacks.csv.
KNOWN_CLEAN_IDS = {50001, 50002, 50004, 50005, 50007, 50009, 50010, 50012,
                   50016, 50017, 50018, 50020}


@pytest.fixture(scope="module")
def pipeline_results():
    """Load real CSVs once and run the full pipeline. Shared across all tests
    in this module so the file I/O happens only once."""
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    return scored, summary, chargebacks


# ── Risk score validity ───────────────────────────────────────────────────────

def test_all_scores_are_integers_in_bounds(pipeline_results):
    scored, _, _ = pipeline_results
    assert scored["risk_score"].between(0, 100).all(), \
        "Every risk_score must be in [0, 100]"


def test_all_risk_labels_are_valid(pipeline_results):
    scored, _, _ = pipeline_results
    unexpected = set(scored["risk_label"]) - {"high", "medium", "low"}
    assert not unexpected, f"Unexpected risk_label values: {unexpected}"


def test_no_null_scores_or_labels(pipeline_results):
    scored, _, _ = pipeline_results
    assert scored["risk_score"].notna().all(), "risk_score must never be null"
    assert scored["risk_label"].notna().all(), "risk_label must never be null"


# ── Known fraud is correctly captured ────────────────────────────────────────

def test_all_known_chargebacks_score_high(pipeline_results):
    """Every confirmed chargeback must land in the high-risk bucket.
    A failure here means the scorer would hide confirmed fraud from the
    review queue, directly increasing undetected fraud losses."""
    scored, _, _ = pipeline_results
    fraud_rows = scored[scored["transaction_id"].isin(KNOWN_CHARGEBACK_IDS)]
    missed = fraud_rows[fraud_rows["risk_label"] != "high"]
    assert missed.empty, (
        f"Known fraud transactions not scored high:\n"
        f"{missed[['transaction_id', 'risk_score', 'risk_label']].to_string(index=False)}"
    )


def test_no_fraud_dollars_in_low_bucket(pipeline_results):
    """Zero confirmed fraud loss should appear in the low-risk bucket.
    If this fails, the fraud team's review queue will not surface the loss."""
    scored, summary, chargebacks = pipeline_results
    low_cb = scored[
        scored["transaction_id"].isin(chargebacks["transaction_id"]) &
        (scored["risk_label"] == "low")
    ]
    total_missed = low_cb["amount_usd"].sum()
    assert total_missed == 0, (
        f"${total_missed:,.2f} of confirmed fraud is sitting in the low-risk bucket"
    )


def test_high_bucket_chargeback_rate_is_one(pipeline_results):
    """All transactions in the high bucket should be confirmed chargebacks —
    chargeback_rate == 1.0 means no false positives in the highest alert tier."""
    _, summary, _ = pipeline_results
    high = summary[summary["risk_label"] == "high"]
    assert not high.empty, "high-risk bucket must exist"
    assert high.iloc[0]["chargeback_rate"] == 1.0


# ── Known clean transactions stay out of high ─────────────────────────────────

def test_known_clean_transactions_not_in_high(pipeline_results):
    """Transactions with no matching chargeback must not appear in the high bucket.
    This guards against false positives that would overload the review queue."""
    scored, _, chargebacks = pipeline_results
    clean_in_high = scored[
        scored["transaction_id"].isin(KNOWN_CLEAN_IDS) &
        (scored["risk_label"] == "high")
    ]
    assert clean_in_high.empty, (
        f"Clean transactions incorrectly scored high:\n"
        f"{clean_in_high[['transaction_id', 'risk_score', 'risk_label']].to_string(index=False)}"
    )


# ── Summary report correctness ────────────────────────────────────────────────

def test_summary_sorted_high_medium_low(pipeline_results):
    _, summary, _ = pipeline_results
    labels = list(summary["risk_label"])
    expected = [l for l in ("high", "medium", "low") if l in labels]
    assert labels == expected, f"Summary not in risk order: {labels}"


def test_summary_chargeback_rates_are_valid_proportions(pipeline_results):
    _, summary, _ = pipeline_results
    for _, row in summary.iterrows():
        rate = row["chargeback_rate"]
        assert 0.0 <= rate <= 1.0, \
            f"chargeback_rate out of range for {row['risk_label']}: {rate}"


def test_summary_transaction_counts_sum_to_total(pipeline_results):
    scored, summary, _ = pipeline_results
    assert summary["transactions"].sum() == len(scored), \
        "Summary transaction counts must add up to the full scored dataset"


def test_summary_amount_totals_sum_to_total(pipeline_results):
    scored, summary, _ = pipeline_results
    assert abs(summary["total_amount_usd"].sum() - scored["amount_usd"].sum()) < 0.01, \
        "Summary amount totals must add up to the full scored dataset"


def test_summary_covers_all_three_risk_labels(pipeline_results):
    _, summary, _ = pipeline_results
    assert set(summary["risk_label"]) == {"high", "medium", "low"}, \
        "All three risk buckets must be populated with the current dataset"
