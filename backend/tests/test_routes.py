"""tests/test_routes.py — _drive_graph's completion-branching logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_drive_graph_offers_approval_when_supplier_recommended():
    from src.api.routes import _drive_graph

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "report_markdown": "## Report",
            "evaluated_suppliers": [{"supplier_id": "SUP-B", "is_recommended": True}],
        }
    )
    mock_db = MagicMock()
    with (
        patch("src.api.routes.get_manager_graph", return_value=mock_graph),
        patch("src.api.routes.db", mock_db),
        patch("src.api.routes.push_event", new=AsyncMock()) as mock_push,
        patch("src.api.routes.end_stream", new=AsyncMock()),
    ):
        await _drive_graph("s1", {"session_id": "s1"})

    event_types = [call.args[1] for call in mock_push.call_args_list]
    assert "approve_ready" in event_types
    assert "completed" not in event_types
    mock_db.update_evaluation.assert_called_once()
    assert mock_db.update_evaluation.call_args.kwargs["status"] == "AWAITING_APPROVAL"


@pytest.mark.asyncio
async def test_drive_graph_completes_without_approval_when_no_evaluation_needed():
    """Regression: a live run against real Supabase hit a 500 from /approve
    ("No recommended supplier found in state") because the old code always pushed
    approve_ready, even when the supervisor correctly skipped Sourcing/Evaluation
    (stock already sufficient) and there was nothing to approve."""
    from src.api.routes import _drive_graph

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={"report_markdown": "## Report", "evaluated_suppliers": None}
    )
    mock_db = MagicMock()
    with (
        patch("src.api.routes.get_manager_graph", return_value=mock_graph),
        patch("src.api.routes.db", mock_db),
        patch("src.api.routes.push_event", new=AsyncMock()) as mock_push,
        patch("src.api.routes.end_stream", new=AsyncMock()),
    ):
        await _drive_graph("s1", {"session_id": "s1"})

    event_types = [call.args[1] for call in mock_push.call_args_list]
    assert "completed" in event_types
    assert "approve_ready" not in event_types
    assert mock_db.update_evaluation.call_args.kwargs["status"] == "COMPLETED"
