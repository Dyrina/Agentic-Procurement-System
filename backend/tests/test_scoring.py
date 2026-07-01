"""
tests/test_scoring.py — Unit tests for the deterministic scoring engine.
"""

from src.services.scoring import (
    DEFAULT_WEIGHTS,
    ScoringWeights,
    rank_suppliers,
    score_supplier,
)


class TestScoreSupplier:
    """Tests for the single-supplier scoring function."""

    def test_returns_float_in_range(self):
        score = score_supplier(
            quote={"quoted_unit_price_sen": 400000, "quoted_delivery_days": 5},
            db_metrics={
                "reliability_score": 90,
                "avg_delivery_days_history": 7,
                "payment_terms": "Net-30",
            },
            historical_avg_price_sen=380000,
        )
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_lower_price_yields_higher_score(self):
        base = {
            "reliability_score": 80,
            "avg_delivery_days_history": 10,
            "payment_terms": "Net-30",
        }
        cheap = score_supplier(
            quote={"quoted_unit_price_sen": 300000, "quoted_delivery_days": 5},
            db_metrics=base,
            historical_avg_price_sen=400000,
        )
        expensive = score_supplier(
            quote={"quoted_unit_price_sen": 500000, "quoted_delivery_days": 5},
            db_metrics=base,
            historical_avg_price_sen=400000,
        )
        assert cheap > expensive

    def test_higher_reliability_yields_higher_score(self):
        quote = {"quoted_unit_price_sen": 400000, "quoted_delivery_days": 5}
        reliable = score_supplier(
            quote=quote,
            db_metrics={
                "reliability_score": 99,
                "avg_delivery_days_history": 10,
                "payment_terms": "Net-30",
            },
            historical_avg_price_sen=400000,
        )
        unreliable = score_supplier(
            quote=quote,
            db_metrics={
                "reliability_score": 30,
                "avg_delivery_days_history": 10,
                "payment_terms": "Net-30",
            },
            historical_avg_price_sen=400000,
        )
        assert reliable > unreliable

    def test_zero_historical_price_gives_neutral(self):
        score = score_supplier(
            quote={"quoted_unit_price_sen": 400000, "quoted_delivery_days": 5},
            db_metrics={
                "reliability_score": 80,
                "avg_delivery_days_history": 10,
                "payment_terms": "Net-30",
            },
            historical_avg_price_sen=0,
        )
        assert 0 <= score <= 100

    def test_custom_weights(self):
        w = ScoringWeights(price=0.7, reliability=0.1, delivery=0.1, terms=0.1)
        score = score_supplier(
            quote={"quoted_unit_price_sen": 200000, "quoted_delivery_days": 5},
            db_metrics={
                "reliability_score": 50,
                "avg_delivery_days_history": 10,
                "payment_terms": "Net-30",
            },
            historical_avg_price_sen=400000,
            weights=w,
        )
        assert 0 <= score <= 100


class TestRankSuppliers:
    """Tests for the multi-supplier ranking function."""

    def test_returns_sorted_list(self):
        quotes = [
            {
                "supplier_id": "SUP-A",
                "supplier_name": "A",
                "quoted_unit_price_sen": 500000,
                "quoted_delivery_days": 14,
            },
            {
                "supplier_id": "SUP-B",
                "supplier_name": "B",
                "quoted_unit_price_sen": 350000,
                "quoted_delivery_days": 3,
            },
        ]
        metrics = {
            "SUP-A": {
                "reliability_score": 70,
                "avg_delivery_days_history": 14,
                "payment_terms": "Net-30",
            },
            "SUP-B": {
                "reliability_score": 95,
                "avg_delivery_days_history": 7,
                "payment_terms": "Net-60",
            },
        }
        ranked = rank_suppliers(quotes, metrics, historical_avg_price_sen=400000)
        assert len(ranked) == 2
        # Should be sorted descending by score
        assert ranked[0]["ai_tradeoff_score"] >= ranked[1]["ai_tradeoff_score"]

    def test_first_is_recommended(self):
        quotes = [
            {
                "supplier_id": "S1",
                "supplier_name": "S1",
                "quoted_unit_price_sen": 100,
                "quoted_delivery_days": 1,
            },
        ]
        ranked = rank_suppliers(quotes, {"S1": {"reliability_score": 90}})
        assert ranked[0]["is_recommended"] is True

    def test_flags_low_reliability(self):
        quotes = [
            {
                "supplier_id": "S1",
                "supplier_name": "S1",
                "quoted_unit_price_sen": 100,
                "quoted_delivery_days": 1,
            },
        ]
        ranked = rank_suppliers(
            quotes, {"S1": {"reliability_score": 50}}, historical_avg_price_sen=100
        )
        risks = ranked[0]["flagged_risks"]
        assert any("reliability" in r.lower() for r in risks)


class TestScoringWeights:
    """Tests for weight validation."""

    def test_default_weights_sum_to_one(self):
        w = DEFAULT_WEIGHTS
        total = w.price + w.reliability + w.delivery + w.terms
        assert abs(total - 1.0) < 1e-6

    def test_bad_weights_raise(self):
        import pytest

        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(price=0.5, reliability=0.5, delivery=0.5, terms=0.5)
