import pytest
from src.services.scoring import (
    min_max_invert,
    payment_terms_score,
    score_suppliers,
)


def test_min_max_invert_lower_is_better():
    # values = [410000, 395000]; lower price → higher score
    assert min_max_invert([410000, 395000], 395000) == pytest.approx(100.0)
    assert min_max_invert([410000, 395000], 410000) == pytest.approx(0.0)


def test_min_max_invert_all_same():
    # Avoid division by zero — return 100
    assert min_max_invert([400000, 400000], 400000) == pytest.approx(100.0)


def test_payment_terms_score_known_values():
    assert payment_terms_score("Net-60") == 100
    assert payment_terms_score("Net-45") == 75
    assert payment_terms_score("Net-30") == 50
    assert payment_terms_score("Net-15") == 25
    assert payment_terms_score("Immediate") == 0
    assert payment_terms_score("Unknown terms") == 50  # default


def test_payment_terms_score_case_insensitive():
    assert payment_terms_score("net-60") == 100
    assert payment_terms_score("NET-30") == 50


def test_score_suppliers_ranking():
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 410000,
            "quoted_delivery_days": 5,
            "payment_terms": "Net-30",
        },
        {
            "supplier_id": "SUP-B",
            "supplier_name": "Global IT",
            "unit_price_sen": 395000,
            "quoted_delivery_days": 2,
            "payment_terms": "Net-60",
        },
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)

    sup_b = next(r for r in results if r["supplier_id"] == "SUP-B")
    sup_a = next(r for r in results if r["supplier_id"] == "SUP-A")

    assert sup_b["total_score"] > sup_a["total_score"]
    assert sup_b["is_recommended"] is True
    assert sup_a["is_recommended"] is False


def test_score_suppliers_risk_flag_price():
    # SUP-A price 410000, avg 365000 → 410000/365000 = 1.123 → >10% → flag
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 410000,
            "quoted_delivery_days": 5,
            "payment_terms": "Net-30",
        }
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)
    flags = results[0]["risk_flags"]
    assert any("Price" in f and "above historical" in f for f in flags)


def test_score_suppliers_no_risk_flag_below_threshold():
    # Price exactly at 1.05x avg — below 10% threshold
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 383250,  # 365000 * 1.05
            "quoted_delivery_days": 7,
            "payment_terms": "Net-30",
        }
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)
    assert results[0]["risk_flags"] == []


def test_score_suppliers_risk_flag_delivery():
    # Delivery 12 days, avg 7 → 12/7 = 1.71 → >10% → flag
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 365000,
            "quoted_delivery_days": 12,
            "payment_terms": "Net-30",
        }
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)
    flags = results[0]["risk_flags"]
    assert any("Delivery" in f and "above historical" in f for f in flags)
