"""
api/routes.py — FastAPI chat endpoints for the agentic procurement system.

POST  /api/v1/chat                       — start a procurement session
GET   /api/v1/chat/{session_id}/stream   — SSE stream of agent updates
POST  /api/v1/chat/{session_id}/approve  — trigger Automation Agent post-approval
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.manager import manager_pipeline
from src.agents.tools.automation import run_automation
from src.api.sse import create_session, end_stream, get_queue, push_event
from src.database.client import SupabaseRepository

router = APIRouter(prefix="/api/v1", tags=["chat"])

db = SupabaseRepository()


# ── Request / Response models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"


class ChatCreated(BaseModel):
    session_id: str


class ApproveResponse(BaseModel):
    status: str
    po_pdf_url: str
    po_number: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatCreated, status_code=202)
async def create_chat(body: ChatRequest) -> ChatCreated:
    """
    Start a new procurement session. Returns session_id immediately.
    Connect to /chat/{session_id}/stream to receive SSE updates.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session_start_ts = datetime.now(timezone.utc).isoformat()

    # Parse item_name and requested_qty from the message
    # The Manager Agent will do the deep parsing — we seed minimal state here
    initial_state = {
        "session_id": session_id,
        "user_id": body.user_id,
        "user_message": body.message,
        "session_start_ts": session_start_ts,
        "plan_attempts": 0,
        "item_name": _extract_item_name(body.message),
        "requested_qty": _extract_quantity(body.message),
    }

    # Create SSE queue and evaluation row before starting background task
    create_session(session_id)
    db.create_evaluation(session_id, body.user_id)

    asyncio.create_task(_run_pipeline(session_id, initial_state))
    return ChatCreated(session_id=session_id)


@router.get("/chat/{session_id}/stream")
async def stream_chat(session_id: str):
    """SSE stream for a procurement session. Connect immediately after POST /chat."""
    q = get_queue(session_id)
    if q is None:
        # Session may have already completed — check DB
        record = db.get_evaluation(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # Stream a synthetic done event
        async def _replay():
            if record.get("report_markdown"):
                yield {"event": "report", "data": json.dumps({"markdown": record["report_markdown"]})}
                yield {"event": "approve_ready", "data": json.dumps({"session_id": session_id})}
        return EventSourceResponse(_replay())

    async def _generator():
        while True:
            event = await q.get()
            if event is None:
                break
            yield {"event": event["type"], "data": json.dumps(event["data"])}

    return EventSourceResponse(_generator())


@router.post("/chat/{session_id}/approve", response_model=ApproveResponse)
async def approve_chat(session_id: str) -> ApproveResponse:
    """Trigger Automation Agent: generate PO, PDF, upload to Supabase Storage."""
    record = db.get_evaluation(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if record["status"] != "AWAITING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Session status is '{record['status']}', expected 'AWAITING_APPROVAL'",
        )

    state = record.get("state_json") or {}
    try:
        updated_state = await run_automation(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Automation failed: {exc}") from exc

    db.update_evaluation(session_id, status="APPROVED")

    return ApproveResponse(
        status="SUCCESS",
        po_pdf_url=updated_state.get("po_pdf_url", ""),
        po_number=updated_state.get("po_number", ""),
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

async def _run_pipeline(session_id: str, initial_state: dict) -> None:
    """Run the LangGraph manager graph as a background task."""
    try:
        await manager_pipeline.ainvoke(initial_state)
    except Exception as exc:
        await push_event(session_id, "error", {"step": "pipeline", "message": str(exc)})
        await end_stream(session_id)
        db.update_evaluation(session_id, status="FAILED")


def _extract_item_name(message: str) -> str:
    """
    Naive item name extraction from the user message.
    The Manager Agent planning prompt handles proper interpretation,
    but we need a seed value for the state.
    """
    # Strip leading quantity words, return rest as item name
    words = message.split()
    for i, word in enumerate(words):
        if word.isdigit():
            return " ".join(words[i + 1:]) if i + 1 < len(words) else message
    return message


def _extract_quantity(message: str) -> int:
    """Extract first integer from message as requested quantity."""
    for word in message.split():
        cleaned = word.rstrip(".,")
        if cleaned.isdigit():
            return int(cleaned)
    return 1
