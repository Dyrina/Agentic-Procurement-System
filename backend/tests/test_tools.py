"""
tests/test_tools.py — Unit tests for MCP tool handlers (mocked DB/Gmail).
"""

from unittest.mock import MagicMock, patch

from src.agents.tools.quotes import _extract_delivery, _extract_price, extract_quotes


class TestExtractPrice:
    """Tests for quote price extraction from email text."""

    def test_price_with_sen_suffix(self):
        assert _extract_price("unit price: 395000 sen") == 395000

    def test_price_without_suffix(self):
        assert _extract_price("Price: 410000") == 410000

    def test_price_rm_format(self):
        assert _extract_price("Our price is RM 3950") == 395000

    def test_no_price_returns_zero(self):
        assert _extract_price("Thank you for your inquiry") == 0


class TestExtractDelivery:
    """Tests for delivery days extraction from email text."""

    def test_delivery_days_label(self):
        assert _extract_delivery("delivery days: 5") == 5

    def test_delivery_colon(self):
        assert _extract_delivery("Delivery: 7") == 7

    def test_n_days_pattern(self):
        assert _extract_delivery("We can deliver in 3 business days") == 3

    def test_no_delivery_returns_zero(self):
        assert _extract_delivery("We will get back to you") == 0


class TestExtractQuotes:
    """Tests for the full extract_quotes function."""

    def test_parses_structured_replies(self):
        raw = [
            {
                "supplier_id": "SUP-A",
                "supplier_name": "Alpha",
                "replies": [
                    {"body_text": "Price: 400000 sen. Delivery: 5 days."},
                ],
            },
        ]
        result = extract_quotes(raw)
        assert len(result) == 1
        assert result[0]["quoted_unit_price_sen"] == 400000
        assert result[0]["quoted_delivery_days"] == 5

    def test_handles_empty_replies(self):
        raw = [{"supplier_id": "SUP-X", "replies": []}]
        result = extract_quotes(raw)
        assert result[0]["quoted_unit_price_sen"] == 0


class TestCheckStock:
    """Tests for the check_stock tool with mocked DB."""

    @patch("src.agents.tools.stock.SupabaseRepository")
    def test_returns_stock_info(self, mock_repo_cls):
        mock_repo = MagicMock()
        mock_repo.get_item.return_value = {
            "item_id": "IT-XPS-15",
            "name": "Dell XPS 15",
            "current_stock": 4,
        }
        mock_repo_cls.return_value = mock_repo

        from src.agents.tools.stock import check_stock

        result = check_stock("IT-XPS-15")
        assert result["current_stock"] == 4
        assert result["stock_warning"] is True

    @patch("src.agents.tools.stock.SupabaseRepository")
    def test_item_not_found(self, mock_repo_cls):
        mock_repo = MagicMock()
        mock_repo.get_item.return_value = None
        mock_repo_cls.return_value = mock_repo

        from src.agents.tools.stock import check_stock

        result = check_stock("NONEXISTENT")
        assert "error" in result


class TestQueryHistory:
    """Tests for the query_history tool with mocked DB."""

    @patch("src.agents.tools.history.SupabaseRepository")
    def test_returns_history(self, mock_repo_cls):
        mock_repo = MagicMock()
        mock_repo.get_purchase_history.return_value = [
            {"unit_price_sen": 360000, "purchase_date": "2025-08-20", "supplier_id": "SUP-B"},
            {"unit_price_sen": 370000, "purchase_date": "2025-06-15", "supplier_id": "SUP-A"},
        ]
        mock_repo_cls.return_value = mock_repo

        from src.agents.tools.history import query_history

        result = query_history("IT-XPS-15")
        assert result["average_past_price_sen"] == 365000
        assert result["last_supplier_id"] == "SUP-B"

    @patch("src.agents.tools.history.SupabaseRepository")
    def test_empty_history(self, mock_repo_cls):
        mock_repo = MagicMock()
        mock_repo.get_purchase_history.return_value = []
        mock_repo_cls.return_value = mock_repo

        from src.agents.tools.history import query_history

        result = query_history("NEW-ITEM")
        assert result["average_past_price_sen"] == 0
        assert result["records"] == []
