"""
tests/test_sse.py — Tests for SSE queue management and event emission.
"""

import asyncio
import pytest
from src.api.sse import create_session, push_event, end_stream, format_sse


@pytest.mark.asyncio
async def test_push_and_drain_events():
    create_session("sess_test")
    await push_event("sess_test", "step_start", {"step": "check_stock"})
    await push_event("sess_test", "step_done", {"step": "check_stock"})
    await end_stream("sess_test")

    from src.api.sse import _queues
    # queue should be removed after end_stream
    assert "sess_test" not in _queues


@pytest.mark.asyncio
async def test_format_sse_output():
    event = {"type": "plan", "data": {"steps": ["check_stock"]}}
    result = format_sse(event)
    assert result.startswith("event: plan\n")
    assert '"steps"' in result


@pytest.mark.asyncio
async def test_push_to_unknown_session_does_nothing():
    # Should not raise
    await push_event("nonexistent", "step_start", {"step": "x"})
