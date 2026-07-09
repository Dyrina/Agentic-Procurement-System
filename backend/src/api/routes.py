"""
api/routes.py — FastAPI chat endpoints for the agentic procurement system.

POST  /api/v1/chat                       — start a procurement session
GET   /api/v1/chat/{session_id}/stream   — SSE stream of agent updates
POST  /api/v1/chat/{session_id}/reply    — answer a paused session's clarification/confirmation
POST  /api/v1/chat/{session_id}/approve  — trigger Automation Agent post-approval
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.manager import get_manager_graph
from src.agents.tools.automation import run_automation
from src.api.sse import create_session, end_stream, get_queue, push_event
from src.database.client import SupabaseRepository

router = APIRouter(prefix="/api/v1", tags=["chat"])

logger = logging.getLogger(__name__)

db = SupabaseRepository()

# asyncio only keeps a weak reference to tasks — without this set a running graph task can be
# garbage-collected mid-flight.
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# ── Request / Response models ───────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"


class ChatCreated(BaseModel):
    session_id: str


class ReplyRequest(BaseModel):
    # Shape depends on which interrupt is pending (intake_clarification /
    # inventory_candidate_confirm / sourcing_timeout — see the `type` field of the
    # "awaiting_input" SSE event); the frontend
    # contract for each isn't designed yet, so this stays a loose passthrough rather than a
    # premature set of typed models.
    reply: dict[str, Any]


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

    # Intake owns all parsing now — no more naive item_name/requested_qty extraction here.
    initial_state = {
        "session_id": session_id,
        "user_id": body.user_id,
        "user_message": body.message,
        "session_start_ts": session_start_ts,
        "worker_calls": 0,
    }

    # Create SSE queue and evaluation row before starting background task
    create_session(session_id)
    await asyncio.to_thread(db.create_evaluation, session_id, body.user_id)

    _spawn(_drive_graph(session_id, initial_state))
    return ChatCreated(session_id=session_id)


@router.post("/chat/{session_id}/reply")
async def reply_chat(session_id: str, body: ReplyRequest) -> dict[str, str]:
    """
    Answer a paused session's clarification/confirmation/escalation. The SSE stream opened by
    /chat stays connected across the pause — this just resumes the same graph run.
    """
    record = await asyncio.to_thread(db.get_evaluation, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if record["status"] != "AWAITING_INPUT":
        raise HTTPException(
            status_code=400,
            detail=f"Session status is '{record['status']}', expected 'AWAITING_INPUT'",
        )

    # Atomic status claim so a double-submitted reply can't resume the same graph twice.
    claimed = await asyncio.to_thread(db.claim_status, session_id, "AWAITING_INPUT", "PLANNING")
    if not claimed:
        raise HTTPException(status_code=409, detail="Session was already resumed")

    _spawn(_drive_graph(session_id, Command(resume=body.reply)))
    return {"status": "RESUMED"}


@router.get("/chat/{session_id}/stream")
async def stream_chat(session_id: str):
    """SSE stream for a procurement session. Connect immediately after POST /chat."""
    q = get_queue(session_id)
    if q is None:
        # Session may have already completed — check DB
        record = await asyncio.to_thread(db.get_evaluation, session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")

        # Stream a synthetic replay event — covers both "already completed" and "the server
        # restarted while this session was paused" (the in-memory SSE queue doesn't survive
        # a restart, but status/awaiting_input_json/report_markdown are persisted).
        async def _replay():
            if record["status"] == "AWAITING_INPUT" and record.get("awaiting_input_json"):
                yield {"event": "awaiting_input", "data": json.dumps(record["awaiting_input_json"])}
                return
            if record.get("report_markdown"):
                yield {
                    "event": "report",
                    "data": json.dumps({"markdown": record["report_markdown"]}),
                }
            if record["status"] == "AWAITING_APPROVAL":
                yield {"event": "approve_ready", "data": json.dumps({"session_id": session_id})}
            elif record["status"] in ("COMPLETED", "CANCELLED"):
                message = (
                    "Session cancelled." if record["status"] == "CANCELLED" else "Session completed."
                )
                yield {
                    "event": "completed",
                    "data": json.dumps({"session_id": session_id, "message": message}),
                }

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
    record = await asyncio.to_thread(db.get_evaluation, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if record["status"] != "AWAITING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Session status is '{record['status']}', expected 'AWAITING_APPROVAL'",
        )

    # Atomic status claim so a double-clicked Approve can't generate two purchase orders.
    claimed = await asyncio.to_thread(db.claim_status, session_id, "AWAITING_APPROVAL", "APPROVING")
    if not claimed:
        raise HTTPException(status_code=409, detail="Approval already in progress")

    state = record.get("state_json") or {}
    try:
        updated_state = await run_automation(state)
    except Exception as exc:
        logger.exception("automation failed for %s", session_id)
        # Release the claim so the user can retry after a transient failure.
        await asyncio.to_thread(db.update_evaluation, session_id, status="AWAITING_APPROVAL")
        raise HTTPException(status_code=500, detail=f"Automation failed: {exc}") from exc

    await asyncio.to_thread(db.update_evaluation, session_id, status="APPROVED")

    return ApproveResponse(
        status="SUCCESS",
        po_pdf_url=updated_state.get("po_pdf_url", ""),
        po_number=updated_state.get("po_number", ""),
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _drive_graph(session_id: str, graph_input: dict | Command) -> None:
    """Run one leg of the manager graph (a fresh start or a resume) as a background task, then
    react to whether it paused (interrupt) or ran to completion."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = await get_manager_graph().ainvoke(graph_input, config=config)
    except Exception as exc:
        logger.exception("graph run crashed for %s", session_id)
        await push_event(session_id, "error", {"step": "pipeline", "message": str(exc)})
        await end_stream(session_id)
        await asyncio.to_thread(db.update_evaluation, session_id, status="FAILED")
        return

    if interrupts := result.get("__interrupt__"):
        payload = interrupts[0].value
        await push_event(session_id, "awaiting_input", payload)
        await asyncio.to_thread(
            db.update_evaluation, session_id, status="AWAITING_INPUT", awaiting_input_json=payload
        )
        return

    # finalize_node/error_node own the terminal status semantics; this just relays them.
    final_status = result.get("status", "COMPLETED")

    if final_status == "FAILED":
        logger.error("graph run failed for %s: %s", session_id, result.get("error"))
        await push_event(
            session_id,
            "error",
            {"step": "pipeline", "message": result.get("error", "Unknown error")},
        )
        await end_stream(session_id)
        await asyncio.to_thread(db.update_evaluation, session_id, status="FAILED")
        return

    if result.get("report_markdown"):
        await push_event(session_id, "report", {"markdown": result["report_markdown"]})

    if final_status == "AWAITING_APPROVAL":
        # Evaluation recommended a supplier — there's actually a PO to generate.
        await push_event(
            session_id,
            "approve_ready",
            {"session_id": session_id, "message": "Approve to generate purchase order"},
        )
    elif final_status == "CANCELLED":
        await push_event(
            session_id,
            "completed",
            {"session_id": session_id, "message": "Session cancelled."},
        )
    else:
        # COMPLETED: check_stock query, satisfied ensure_stock, or out-of-scope rejection.
        message = result.get("completion_message") or (
            "Stock check complete."
            if result.get("intent") == "check_stock"
            else "No purchase order needed — stock is sufficient."
        )
        await push_event(
            session_id, "completed", {"session_id": session_id, "message": message}
        )

    await end_stream(session_id)
    await asyncio.to_thread(
        db.update_evaluation,
        session_id,
        status=final_status,
        report_markdown=result.get("report_markdown", ""),
        state_json=dict(result),
    )
