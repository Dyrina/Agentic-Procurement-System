"""
core/state.py — ProcurementState TypedDict (shared state flowing through all nodes/tools).

This is the single source of truth for the data structure that flows through
the LangGraph manager graph.  Every node reads from and writes to this dict.
"""

from __future__ import annotations
from typing import Any, TypedDict


class ProcurementState(TypedDict, total=False):
    # Session metadata
    session_id: str
    user_id: str
    user_message: str
    session_start_ts: str       # ISO timestamp, used as Gmail search lower bound

    # Supervisor loop
    next_worker: str
    worker_calls: int
    supervisor_history: list[dict[str, str]]   # [{"worker": str, "summary": str}], deterministic scratchpad

    status: str                 # EXECUTING | AWAITING_INPUT | AWAITING_APPROVAL | APPROVED | FAILED
    error: str | None

    # Intake
    intake_attempts: int
    needs_clarification: bool
    clarification_payload: dict[str, Any] | None

    # check_stock outputs
    item_name: str
    item_id: str
    requested_qty: int
    stock_sufficient: bool
    current_stock: int
    inventory_candidates: list[dict[str, Any]] | None

    # send_rfqs outputs
    rfq_sent_at: str            # ISO timestamp
    supplier_emails: list[str]

    # wait_for_quotes outputs
    all_replied: bool
    pending_emails: list[str]

    # extract_quotes outputs
    extracted_quotes: list[dict[str, Any]]
    # each: {supplier_id, supplier_name, unit_price_sen, quoted_delivery_days, payment_terms}

    # evaluate_suppliers outputs
    evaluated_suppliers: list[dict[str, Any]]
    # each: {supplier_id, supplier_name, unit_price_sen, quoted_delivery_days, payment_terms,
    #        total_score, risk_flags, is_recommended, reasoning}

    # generate_report outputs
    report_markdown: str

    # post-approval automation outputs (unchanged)
    po_pdf_url: str
    po_number: str
