import pandas as pd

from analyze_fraud import summarize_results


def _scored(*rows):
    return pd.DataFrame(rows)


def _chargebacks(*ids):
    return pd.DataFrame({"transaction_id": list(ids)})


def test_summarize_results_sort_order():
    scored = _scored(
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "low"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "medium"},
        {"transaction_id": 3, "amount_usd": 300.0, "risk_label": "high"},
    )
    summary = summarize_results(scored, _chargebacks())
    assert list(summary["risk_label"]) == ["high", "medium", "low"]


def test_summarize_results_chargeback_rate_all_fraud():
    scored = _scored(
        {"transaction_id": 1, "amount_usd": 500.0, "risk_label": "high"},
        {"transaction_id": 2, "amount_usd": 400.0, "risk_label": "high"},
    )
    summary = summarize_results(scored, _chargebacks(1, 2))
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == 1.0
    assert row["chargebacks"] == 2


def test_summarize_results_chargeback_rate_no_fraud():
    scored = _scored(
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "low"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "low"},
    )
    summary = summarize_results(scored, _chargebacks())
    row = summary[summary["risk_label"] == "low"].iloc[0]
    assert row["chargeback_rate"] == 0.0
    assert row["chargebacks"] == 0


def test_summarize_results_chargeback_rate_partial():
    scored = _scored(
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "medium"},
        {"transaction_id": 2, "amount_usd": 100.0, "risk_label": "medium"},
        {"transaction_id": 3, "amount_usd": 100.0, "risk_label": "medium"},
        {"transaction_id": 4, "amount_usd": 100.0, "risk_label": "medium"},
    )
    summary = summarize_results(scored, _chargebacks(1, 2))
    row = summary[summary["risk_label"] == "medium"].iloc[0]
    assert row["chargeback_rate"] == 0.5
    assert row["chargebacks"] == 2


def test_summarize_results_totals():
    scored = _scored(
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "high"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "high"},
        {"transaction_id": 3, "amount_usd": 50.0,  "risk_label": "low"},
    )
    summary = summarize_results(scored, _chargebacks(1))
    high = summary[summary["risk_label"] == "high"].iloc[0]
    assert high["transactions"] == 2
    assert high["total_amount_usd"] == 300.0
    assert high["avg_amount_usd"] == 150.0


def test_summarize_results_chargeback_rate_between_zero_and_one():
    """Chargeback rate must always be a valid proportion."""
    scored = _scored(
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "high"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "medium"},
        {"transaction_id": 3, "amount_usd": 50.0,  "risk_label": "low"},
    )
    summary = summarize_results(scored, _chargebacks(1))
    for rate in summary["chargeback_rate"]:
        assert 0.0 <= rate <= 1.0, f"chargeback_rate out of range: {rate}"
