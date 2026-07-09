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
            "status": "AWAITING_APPROVAL",
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
        return_value={
            "status": "COMPLETED",
            "report_markdown": "## Report",
            "evaluated_suppliers": None,
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
    assert "completed" in event_types
    assert "approve_ready" not in event_types
    assert mock_db.update_evaluation.call_args.kwargs["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_drive_graph_relays_cancelled_status():
    from src.api.routes import _drive_graph

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"status": "CANCELLED", "cancelled": True})
    mock_db = MagicMock()
    with (
        patch("src.api.routes.get_manager_graph", return_value=mock_graph),
        patch("src.api.routes.db", mock_db),
        patch("src.api.routes.push_event", new=AsyncMock()) as mock_push,
        patch("src.api.routes.end_stream", new=AsyncMock()),
    ):
        await _drive_graph("s1", {"session_id": "s1"})

    event_types = [call.args[1] for call in mock_push.call_args_list]
    assert event_types == ["completed"]
    assert "cancelled" in mock_push.call_args_list[0].args[2]["message"].lower()
    assert mock_db.update_evaluation.call_args.kwargs["status"] == "CANCELLED"


@pytest.mark.asyncio
async def test_reply_returns_409_when_status_claim_lost():
    """Two rapid replies race: only the one that wins the atomic status claim may resume."""
    from fastapi import HTTPException

    from src.api.routes import ReplyRequest, reply_chat

    mock_db = MagicMock()
    mock_db.get_evaluation.return_value = {"status": "AWAITING_INPUT"}
    mock_db.claim_status.return_value = False
    with patch("src.api.routes.db", mock_db):
        with pytest.raises(HTTPException) as excinfo:
            await reply_chat("s1", ReplyRequest(reply={"text": "hi"}))

    assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_approve_returns_409_when_status_claim_lost():
    """A double-clicked Approve must not run automation (and generate a PO) twice."""
    from fastapi import HTTPException

    from src.api.routes import approve_chat

    mock_db = MagicMock()
    mock_db.get_evaluation.return_value = {"status": "AWAITING_APPROVAL", "state_json": {}}
    mock_db.claim_status.return_value = False
    with (
        patch("src.api.routes.db", mock_db),
        patch("src.api.routes.run_automation", new=AsyncMock()) as mock_automation,
    ):
        with pytest.raises(HTTPException) as excinfo:
            await approve_chat("s1")

    assert excinfo.value.status_code == 409
    mock_automation.assert_not_called()
