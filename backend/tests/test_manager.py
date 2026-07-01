"""
tests/test_manager.py — Integration test for the manager graph flow.
"""

from unittest.mock import MagicMock, patch, AsyncMock


import pytest

class TestManagerGraph:
    """Tests for the LangGraph manager pipeline."""

    @pytest.mark.asyncio
    @patch("src.agents.manager.ChatGoogleGenerativeAI")
    @patch("src.agents.tools.stock.SupabaseRepository")
    @patch("src.agents.tools.history.SupabaseRepository")
    @patch("src.agents.manager.SupabaseRepository")
    async def test_full_pipeline_produces_final_payload(
        self, mock_manager_repo, mock_hist_repo, mock_stock_repo, mock_llm_class
    ):
        """The pipeline should produce a final_payload with status."""
        # Mock stock
        stock_mock = MagicMock()
        stock_mock.select.return_value = [
            {
                "item_id": "IT-XPS-15",
                "name": "Dell XPS 15",
                "current_stock": 4,
            }
        ]
        mock_stock_repo.return_value = stock_mock

        # Mock history
        hist_mock = MagicMock()
        hist_mock.get_purchase_history.return_value = [
            {"unit_price_sen": 365000, "purchase_date": "2025-08-20", "supplier_id": "SUP-B", "delivery_days": 7},
        ]
        mock_hist_repo.return_value = hist_mock

        # Mock manager DB updates
        manager_mock = MagicMock()
        mock_manager_repo.return_value = manager_mock

        # Mock LLM response for the plan node
        mock_llm_instance = MagicMock()
        mock_structured = AsyncMock()
        mock_plan_output = MagicMock()
        mock_plan_output.plan = ["check_stock", "query_history", "evaluate_suppliers", "generate_report"]
        mock_structured.ainvoke.return_value = mock_plan_output
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_llm_class.return_value = mock_llm_instance

        from src.agents.manager import manager_graph

        initial_state = {
            "session_id": "eval_test001",
            "user_message": "Evaluate suppliers for IT-XPS-15",
            "item_name": "Dell XPS 15",
            "requested_qty": 30,
            "extracted_quotes": [
                {
                    "supplier_id": "SUP-B",
                    "supplier_name": "Global IT",
                    "unit_price_sen": 395000,
                    "quoted_delivery_days": 2,
                    "payment_terms": "Net-60"
                }
            ],
            "status": "PROCESSING",
        }

        result = await manager_graph.ainvoke(initial_state)

        assert result.get("evaluated_suppliers") is not None
        assert result["session_id"] == "eval_test001"
        assert result["status"] == "AWAITING_APPROVAL"

