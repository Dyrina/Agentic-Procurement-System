"""
tests/test_manager.py — Integration test for the manager graph flow.
"""

from unittest.mock import MagicMock, patch


class TestManagerGraph:
    """Tests for the LangGraph manager pipeline."""

    @patch("src.agents.tools.stock.SupabaseRepository")
    @patch("src.agents.tools.history.SupabaseRepository")
    @patch("src.agents.tools.evaluation.SupabaseRepository")
    def test_full_pipeline_produces_final_payload(
        self, mock_eval_repo, mock_hist_repo, mock_stock_repo
    ):
        """The pipeline should produce a final_payload with status."""
        # Mock stock
        stock_mock = MagicMock()
        stock_mock.get_item.return_value = {
            "item_id": "IT-XPS-15",
            "name": "Dell XPS 15",
            "current_stock": 4,
        }
        mock_stock_repo.return_value = stock_mock

        # Mock history
        hist_mock = MagicMock()
        hist_mock.get_purchase_history.return_value = [
            {"unit_price_sen": 365000, "purchase_date": "2025-08-20", "supplier_id": "SUP-B"},
        ]
        mock_hist_repo.return_value = hist_mock

        # Mock evaluation DB lookup
        eval_mock = MagicMock()
        eval_mock.get_supplier.side_effect = lambda sid: {
            "SUP-A": {
                "reliability_score": 85,
                "avg_delivery_days": 14,
                "payment_terms": "Net-30",
                "contact_email": "a@test.com",
            },
            "SUP-B": {
                "reliability_score": 98,
                "avg_delivery_days": 7,
                "payment_terms": "Net-60",
                "contact_email": "b@test.com",
            },
        }.get(sid)
        mock_eval_repo.return_value = eval_mock

        from src.agents.manager import manager_pipeline

        initial_state = {
            "evaluation_id": "eval_test001",
            "request_text": "Evaluate suppliers for IT-XPS-15",
            "pdf_paths": [],
            "status": "PROCESSING",
            "sse_queue": None,
        }

        result = manager_pipeline.invoke(initial_state)

        assert result.get("final_payload") is not None
        assert result["final_payload"]["evaluation_id"] == "eval_test001"
        assert "evaluated_suppliers" in result["final_payload"]
        assert result["status"] in ("AWAITING_MANAGER_APPROVAL", "COMPLETED")

    def test_plan_node_extracts_item_id(self):
        """The plan node should extract item IDs from request text."""
        from src.agents.manager import _extract_item_id

        assert _extract_item_id("Need 30 units of IT-XPS-15") == "IT-XPS-15"
        assert _extract_item_id("Order OF-CHAIR-E") == "OF-CHAIR-E"
        assert _extract_item_id("no item here") == "IT-XPS-15"  # default fallback
