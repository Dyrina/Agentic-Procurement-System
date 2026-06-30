"""
agents/graph.py — LangGraph sequential state machine for procurement evaluation.

Defines:
  • ProcurementState — the shared TypedDict flowing through every node.
  • Four placeholder agent nodes (document → analyst → evaluation → reporting).
  • A compiled LangGraph ``StateGraph`` wired linearly to END.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


# ── Shared state flowing through the pipeline ──────────────────────────────


class ProcurementState(TypedDict, total=False):
    """State dictionary threaded through every node in the graph."""

    # Inputs
    evaluation_id: str
    request_text: str
    pdf_paths: list[str]

    # Populated by document_agent
    extracted_quotes: list[dict[str, Any]]

    # Populated by supabase_analyst
    historical_data: dict[str, Any]

    # Populated by evaluation_agent
    evaluated_suppliers: list[dict[str, Any]]

    # Populated by reporting_agent
    final_payload: dict[str, Any]

    # Pipeline metadata
    current_node: str
    status: str


# ── Agent nodes (mock implementations) ─────────────────────────────────────


def document_agent(state: ProcurementState) -> ProcurementState:
    """
    Node 1 — Document Agent.

    Responsible for parsing uploaded PDFs / raw text and extracting
    structured quote data.  Currently returns mock extracted quotes.
    """
    state["current_node"] = "document_agent"
    state["extracted_quotes"] = [
        {
            "supplier_id": "SUP-B",
            "supplier_name": "Global IT Supplies",
            "quoted_unit_price_sen": 395000,
            "quoted_delivery_days": 2,
        },
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech Solutions",
            "quoted_unit_price_sen": 410000,
            "quoted_delivery_days": 5,
        },
    ]
    return state


def supabase_analyst(state: ProcurementState) -> ProcurementState:
    """
    Node 2 — Supabase Analyst.

    Queries the Supabase database for inventory levels, purchase history,
    and supplier metrics.  Currently returns mock historical context.
    """
    state["current_node"] = "supabase_analyst"
    state["historical_data"] = {
        "material_context": {
            "item_id": "IT-XPS-15",
            "item_name": "Dell XPS 15 Laptop",
            "requested_qty": 30,
            "current_stock": 4,
            "stock_warning": True,
        },
        "historical_context": {
            "average_past_price_sen": 365000,
            "last_purchase_date": "2025-08-20",
            "last_supplier_id": "SUP-B",
        },
        "supplier_db_metrics": {
            "SUP-B": {
                "reliability_score": 98,
                "avg_delivery_days_history": 7,
                "payment_terms": "Net-60",
                "contact_email": "sales@globalit.com",
            },
            "SUP-A": {
                "reliability_score": 85,
                "avg_delivery_days_history": 14,
                "payment_terms": "Net-30",
                "contact_email": "orders@alphatech.com",
            },
        },
    }
    return state


def evaluation_agent(state: ProcurementState) -> ProcurementState:
    """
    Node 3 — Evaluation Agent.

    Merges quoted data with database metrics and computes an AI trade-off
    score for each supplier.  Currently returns mock evaluated suppliers.
    """
    state["current_node"] = "evaluation_agent"

    db_metrics = state.get("historical_data", {}).get("supplier_db_metrics", {})
    quotes = state.get("extracted_quotes", [])

    evaluated: list[dict[str, Any]] = []
    for quote in quotes:
        sid = quote["supplier_id"]
        metrics = db_metrics.get(sid, {})
        is_recommended = sid == "SUP-B"
        evaluated.append(
            {
                "supplier_id": sid,
                "supplier_name": quote["supplier_name"],
                "database_metrics": {
                    "reliability_score": metrics.get("reliability_score", 0),
                    "avg_delivery_days_history": metrics.get("avg_delivery_days_history", 0),
                    "payment_terms": metrics.get("payment_terms", "N/A"),
                    "contact_email": metrics.get("contact_email", ""),
                },
                "quoted_metrics": {
                    "quoted_unit_price_sen": quote["quoted_unit_price_sen"],
                    "quoted_delivery_days": quote["quoted_delivery_days"],
                },
                "ai_tradeoff_score": 92.5 if is_recommended else 74.0,
                "is_recommended": is_recommended,
                "flagged_risks": [] if is_recommended else ["Higher unit cost"],
            }
        )

    state["evaluated_suppliers"] = evaluated
    return state


def reporting_agent(state: ProcurementState) -> ProcurementState:
    """
    Node 4 — Reporting Agent.

    Assembles the final evaluation payload (matching the Output Contract)
    and prepares an executive summary.  Currently returns mock output.
    """
    state["current_node"] = "reporting_agent"

    historical = state.get("historical_data", {})
    material = historical.get("material_context", {})
    hist_ctx = historical.get("historical_context", {})
    suppliers = state.get("evaluated_suppliers", [])

    recommended = next((s for s in suppliers if s.get("is_recommended")), None)
    rec_name = recommended["supplier_name"] if recommended else "N/A"

    summary_md = (
        f"## Procurement Evaluation Summary\n\n"
        f"**Item:** {material.get('item_name', 'N/A')}  \n"
        f"**Requested Qty:** {material.get('requested_qty', 0)}  \n"
        f"**Current Stock:** {material.get('current_stock', 0)}  \n\n"
        f"### Recommendation\n\n"
        f"**{rec_name}** is the recommended supplier based on the highest "
        f"AI trade-off score, factoring in reliability, delivery speed, "
        f"and quoted pricing.\n\n"
        f"*This evaluation was performed automatically by the AI Procurement "
        f"Operations Multi-Agent System.*"
    )

    state["final_payload"] = {
        "evaluation_id": state.get("evaluation_id", "eval_unknown"),
        "status": "AWAITING_MANAGER_APPROVAL",
        "material_context": material,
        "historical_context": hist_ctx,
        "evaluated_suppliers": suppliers,
        "reporting_agent_output": {
            "executive_summary_markdown": summary_md,
            "audit_preview": {
                "action_type": "SUPPLIER_EVALUATION",
                "agents_executed": [
                    "Manager",
                    "Document",
                    "Analyst",
                    "Evaluation",
                    "Reporting",
                ],
            },
        },
    }
    state["status"] = "AWAITING_MANAGER_APPROVAL"
    return state


# ── Graph construction ─────────────────────────────────────────────────────


def build_procurement_graph() -> StateGraph:
    """
    Build and compile the sequential LangGraph pipeline:

        document_agent → supabase_analyst → evaluation_agent → reporting_agent → END
    """
    graph = StateGraph(ProcurementState)

    # Register nodes
    graph.add_node("document_agent", document_agent)
    graph.add_node("supabase_analyst", supabase_analyst)
    graph.add_node("evaluation_agent", evaluation_agent)
    graph.add_node("reporting_agent", reporting_agent)

    # Wire linear edges
    graph.set_entry_point("document_agent")
    graph.add_edge("document_agent", "supabase_analyst")
    graph.add_edge("supabase_analyst", "evaluation_agent")
    graph.add_edge("evaluation_agent", "reporting_agent")
    graph.add_edge("reporting_agent", END)

    return graph.compile()


# Pre-compiled graph instance for import elsewhere
procurement_pipeline = build_procurement_graph()
