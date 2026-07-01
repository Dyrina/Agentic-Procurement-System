from __future__ import annotations

import asyncio
import json
from typing import Any

_queues: dict[str, asyncio.Queue] = {}


def create_session(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _queues[session_id] = q
    return q


def get_queue(session_id: str) -> asyncio.Queue | None:
    return _queues.get(session_id)


async def push_event(session_id: str, event_type: str, data: dict[str, Any]) -> None:
    q = _queues.get(session_id)
    if q:
        await q.put({"type": event_type, "data": data})


async def end_stream(session_id: str) -> None:
    q = _queues.get(session_id)
    if q:
        await q.put(None)  # sentinel
    _queues.pop(session_id, None)


def format_sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
