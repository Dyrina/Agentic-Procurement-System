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

    # Planning
    plan: list[str]
    plan_attempts: int
    plan_error: str | None
    validation_passed: bool

    # Execution
    current_step: str
    status: str                 # PLANNING | EXECUTING | AWAITING_APPROVAL | APPROVED | FAILED
    error: str | None

    # check_stock outputs
    item_name: str
    item_id: str
    requested_qty: int
    stock_sufficient: bool
    current_stock: int

    # send_rfqs outputs
    rfq_sent_at: str            # ISO timestamp
    supplier_emails: list[str]

    # wait_for_quotes outputs
    all_replied: bool

    # extract_quotes outputs
    extracted_quotes: list[dict[str, Any]]
    # each: {supplier_id, supplier_name, unit_price_sen, quoted_delivery_days, payment_terms}

    # query_history outputs
    avg_unit_price_sen: float
    avg_delivery_days: float

    # evaluate_suppliers outputs
    evaluated_suppliers: list[dict[str, Any]]
    # each: {supplier_id, supplier_name, unit_price_sen, quoted_delivery_days, payment_terms,
    #        price_score, delivery_score, payment_terms_score, total_score, risk_flags, is_recommended}

    # generate_report outputs
    report_markdown: str