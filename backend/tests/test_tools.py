"""Unit tests for tool handlers using mocked dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.state import ProcurementState

# ── query_history ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_history_computes_averages():
    from src.agents.tools.history import query_history

    mock_db = MagicMock()
    mock_db.get_purchase_history.return_value = [
        {"unit_price_sen": 365000, "delivery_days": 7},
        {"unit_price_sen": 385000, "delivery_days": 9},
    ]
    with patch("src.agents.tools.history.SupabaseRepository", return_value=mock_db):
        result = await query_history("IT-XPS-15")
        assert result["avg_unit_price_sen"] == pytest.approx(375000.0)
        assert result["avg_delivery_days"] == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_query_history_no_history_returns_zeros():
    from src.agents.tools.history import query_history

    mock_db = MagicMock()
    mock_db.get_purchase_history.return_value = []
    with patch("src.agents.tools.history.SupabaseRepository", return_value=mock_db):
        result = await query_history("IT-NEW")
        assert result["avg_unit_price_sen"] == 0.0
        assert result["avg_delivery_days"] == 0.0


# ── generate_report ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_report_handler_contains_key_fields():
    from src.agents.tools.report import generate_report_handler

    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "Buy 30 laptops",
        "item_name": "Dell XPS 15 Laptop",
        "requested_qty": 30,
        "stock_sufficient": False,
        "current_stock": 4,
        "evaluated_suppliers": [
            {
                "supplier_id": "SUP-B",
                "supplier_name": "Global IT",
                "unit_price_sen": 395000,
                "quoted_delivery_days": 2,
                "payment_terms": "Net-60",
                "price_score": 100.0,
                "delivery_score": 100.0,
                "payment_terms_score": 100.0,
                "total_score": 100.0,
                "risk_flags": [],
                "is_recommended": True,
            }
        ],
    }
    result = await generate_report_handler(state)
    md = result["report_markdown"]
    assert "Global IT" in md
    assert "Dell XPS 15" in md
    assert "Recommended" in md or "recommended" in md
