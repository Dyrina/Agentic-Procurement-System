"""
core/schemas.py — Pydantic models (contracts) for the Procurement API.

Defines the exact Input and Output contracts described in the system spec.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Input Contract ──────────────────────────────────────────────────────────


class AuthContext(BaseModel):
    """Authentication / authorisation context sent with each request."""

    user_id: str
    department: str


class EvaluationRequest(BaseModel):
    """
    Input payload for POST /api/v1/evaluations.

    Sent as a JSON string inside a multipart/form-data request alongside
    optional file uploads.
    """

    request_id: str
    auth_context: AuthContext
    raw_request_text: str


# ── Output Contract (nested models) ────────────────────────────────────────


class MaterialContext(BaseModel):
    item_id: str
    item_name: str
    requested_qty: int
    current_stock: int
    stock_warning: bool


class HistoricalContext(BaseModel):
    average_past_price_sen: int
    last_purchase_date: str
    last_supplier_id: str


class DatabaseMetrics(BaseModel):
    reliability_score: int
    avg_delivery_days_history: int
    payment_terms: str
    contact_email: str


class QuotedMetrics(BaseModel):
    quoted_unit_price_sen: int
    quoted_delivery_days: int


class EvaluatedSupplier(BaseModel):
    supplier_id: str
    supplier_name: str
    database_metrics: DatabaseMetrics
    quoted_metrics: QuotedMetrics
    ai_tradeoff_score: float
    is_recommended: bool
    flagged_risks: list[str] = Field(default_factory=list)


class AuditPreview(BaseModel):
    action_type: str
    agents_executed: list[str]


class ReportingAgentOutput(BaseModel):
    executive_summary_markdown: str
    audit_preview: AuditPreview


class EvaluationResponse(BaseModel):
    """
    Full output contract returned by GET /api/v1/evaluations/{id}.
    Matches the JSON shape specified in the system design document.
    """

    evaluation_id: str
    status: str
    material_context: MaterialContext
    historical_context: HistoricalContext
    evaluated_suppliers: list[EvaluatedSupplier]
    reporting_agent_output: ReportingAgentOutput


# ── Lightweight response models ────────────────────────────────────────────


class EvaluationCreated(BaseModel):
    """Returned immediately after POST /api/v1/evaluations."""

    evaluation_id: str
    status: str = "PROCESSING"


class EvaluationStatus(BaseModel):
    """Returned by GET /api/v1/evaluations/{id}/status."""

    evaluation_id: str
    status: str
    current_node: str | None = None


class ApprovalResponse(BaseModel):
    """Returned by POST /api/v1/evaluations/{id}/approve."""

    status: str
    message: str
