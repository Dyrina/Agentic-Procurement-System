"""
api/routes.py — FastAPI router for the Procurement Evaluation API.

Endpoints:
  POST   /api/v1/evaluations            — create evaluation (multipart/form-data)
  GET    /api/v1/evaluations/{id}/status — poll current pipeline node
  GET    /api/v1/evaluations/{id}        — full evaluation result
  POST   /api/v1/evaluations/{id}/approve — generate purchase order
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.agents.graph import procurement_pipeline
from src.core.schemas import (
    ApprovalResponse,
    EvaluationCreated,
    EvaluationResponse,
    EvaluationStatus,
)
from src.database.client import SupabaseRepository

router = APIRouter(prefix="/api/v1", tags=["evaluations"])

# ── In-memory store (replaced by a real DB / cache in production) ──────────

_evaluations: dict[str, dict[str, Any]] = {}

db = SupabaseRepository()


# ── Helpers ────────────────────────────────────────────────────────────────


async def _run_pipeline(evaluation_id: str, request_text: str, pdf_paths: list[str]) -> None:
    """Execute the LangGraph pipeline in a background task."""
    initial_state = {
        "evaluation_id": evaluation_id,
        "request_text": request_text,
        "pdf_paths": pdf_paths,
        "status": "PROCESSING",
        "current_node": "document_agent",
    }

    # LangGraph's .invoke() is synchronous — run in a thread so we don't
    # block the async event loop.
    result = await asyncio.to_thread(procurement_pipeline.invoke, initial_state)

    # Persist the final payload produced by the reporting_agent.
    _evaluations[evaluation_id] = {
        "status": result.get("status", "COMPLETED"),
        "current_node": result.get("current_node", "reporting_agent"),
        "final_payload": result.get("final_payload", {}),
    }

    # Write an audit log entry to Supabase.
    try:
        db.write_audit_log(
            action_type="SUPPLIER_EVALUATION",
            agent_name="procurement_pipeline",
            decision_json=result.get("final_payload", {}),
        )
    except Exception:
        # Non-critical — don't crash the pipeline if audit write fails.
        pass


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/evaluations", response_model=EvaluationCreated, status_code=202)
async def create_evaluation(
    payload: str = Form(..., description="JSON string matching the EvaluationRequest schema"),
    files: list[UploadFile] = File(default=[]),
) -> EvaluationCreated:
    """
    Accept a multipart/form-data request containing:
      • ``payload``  — a JSON string matching the Input Contract.
      • ``files``    — zero or more uploaded PDF / document files.

    Creates an evaluation record, triggers the LangGraph pipeline as a
    background task, and returns immediately with the evaluation ID.
    """
    # Parse the JSON payload string
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in payload field: {exc}") from exc

    evaluation_id = f"eval_{uuid.uuid4().hex[:6]}"

    # Save uploaded file paths (in production you'd stream to object storage)
    pdf_paths: list[str] = []
    for f in files:
        pdf_paths.append(f.filename or "unnamed.pdf")

    # Seed the in-memory store so /status returns immediately
    _evaluations[evaluation_id] = {
        "status": "PROCESSING",
        "current_node": "document_agent",
        "final_payload": None,
    }

    # Fire-and-forget background pipeline execution
    asyncio.create_task(
        _run_pipeline(
            evaluation_id=evaluation_id,
            request_text=data.get("raw_request_text", ""),
            pdf_paths=pdf_paths,
        )
    )

    return EvaluationCreated(evaluation_id=evaluation_id, status="PROCESSING")


@router.get("/evaluations/{evaluation_id}/status", response_model=EvaluationStatus)
async def get_evaluation_status(evaluation_id: str) -> EvaluationStatus:
    """Return the current pipeline-node execution status for an evaluation."""
    record = _evaluations.get(evaluation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status=record["status"],
        current_node=record.get("current_node"),
    )


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(evaluation_id: str) -> EvaluationResponse:
    """
    Return the full Output Contract for a completed evaluation.

    Returns 404 if the evaluation ID is unknown, or 202 if the pipeline
    is still running.
    """
    record = _evaluations.get(evaluation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    if record["status"] == "PROCESSING" or record.get("final_payload") is None:
        raise HTTPException(status_code=202, detail="Evaluation still processing")

    return EvaluationResponse(**record["final_payload"])


@router.post("/evaluations/{evaluation_id}/approve", response_model=ApprovalResponse)
async def approve_evaluation(evaluation_id: str) -> ApprovalResponse:
    """
    Approve an evaluation: generates a purchase order in Supabase for the
    recommended supplier and updates the evaluation status.
    """
    record = _evaluations.get(evaluation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    final = record.get("final_payload")
    if final is None:
        raise HTTPException(status_code=400, detail="Evaluation not yet completed")

    # Find the recommended supplier
    suppliers = final.get("evaluated_suppliers", [])
    recommended = next((s for s in suppliers if s.get("is_recommended")), None)
    if recommended is None:
        raise HTTPException(status_code=400, detail="No recommended supplier found")

    # Calculate total amount (quoted price × requested qty)
    material = final.get("material_context", {})
    qty = material.get("requested_qty", 1)
    unit_price = recommended.get("quoted_metrics", {}).get("quoted_unit_price_sen", 0)
    total_amount_sen = unit_price * qty

    # Write purchase order to Supabase
    try:
        db.create_purchase_order(
            supplier_id=recommended["supplier_id"],
            total_amount_sen=total_amount_sen,
            approved_by="system",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write purchase order: {exc}",
        ) from exc

    # Update in-memory status
    record["status"] = "APPROVED"

    return ApprovalResponse(status="SUCCESS", message="PO Generated")
