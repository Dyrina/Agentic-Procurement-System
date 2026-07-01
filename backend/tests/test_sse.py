"""
tests/test_sse.py — Tests for SSE queue management and event emission.
"""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_create_and_emit():
    """Creating a session and emitting events should populate the queue."""
    from src.api.sse import create_session, destroy_session, emit

    session_id, queue = create_session()
    assert session_id.startswith("sess_")

    await emit(session_id, "status", {"message": "hello"})
    event = await asyncio.wait_for(queue.get(), timeout=1.0)

    assert event.event == "status"
    assert event.data["message"] == "hello"

    destroy_session(session_id)


@pytest.mark.asyncio
async def test_emit_to_missing_session():
    """Emitting to a nonexistent session should not raise."""
    from src.api.sse import emit

    await emit("nonexistent_session", "test", {"x": 1})
    # Should not raise


@pytest.mark.asyncio
async def test_event_generator_stops_on_done():
    """The event generator should stop after a 'done' event."""
    from src.api.sse import SSEEvent, create_session, event_generator

    _, queue = create_session()
    await queue.put(SSEEvent(event="status", data={"step": 1}))
    await queue.put(SSEEvent(event="done", data={"status": "ok"}))

    events = []
    async for ev in event_generator(queue):
        events.append(ev)

    assert len(events) == 2
    assert events[-1]["event"] == "done"


class TestSSEEventEncode:
    """Tests for SSEEvent serialisation."""

    def test_encode_format(self):
        from src.api.sse import SSEEvent

        ev = SSEEvent(event="plan", data={"steps": 4}, id="123")
        encoded = ev.encode()
        assert "id: 123" in encoded
        assert "event: plan" in encoded
        assert '"steps": 4' in encoded
