"""agents/workers/evaluation.py — Evaluation specialist: judges suppliers, writes an audit trail.

Replaces agents/tools/evaluation.py's forced deterministic weighted formula (price 55% /
delivery 30% / terms 15%) with free LLM judgment — services/scoring.py's score_suppliers is
still available as a reference number the agent can consult, but it no longer dictates the
recommendation. Since the result is no longer reproducible math, every decision's full
reasoning is written to audit_logs, not just the final scores.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from src.agents.tools.history import query_history
from src.agents.workers import _build_llm, _last_tool_call
from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.services.scoring import score_suppliers


class _EvaluatedSupplier(BaseModel):
    supplier_id: str
    supplier_name: str
    unit_price_sen: int
    quoted_delivery_days: int
    payment_terms: str
    total_score: float
    risk_flags: list[str]
    is_recommended: bool
    reasoning: str


@tool
async def get_purchase_history(item_id: str) -> dict:
    """Fetch historical average unit price (sen) and delivery days (in days) for this item."""
    return await query_history(item_id)


@tool
def get_reference_score(
    quotes: list[dict], avg_unit_price_sen: float, avg_delivery_days: float
) -> list[dict]:
    """Compute a deterministic reference score (price 55% / delivery 30% / terms 15%) per
    supplier. This is a reference point, not the final answer — you may weigh things
    differently based on risk context, but explain your reasoning either way."""
    return score_suppliers(quotes, avg_unit_price_sen, avg_delivery_days)


@tool
def write_audit_log(evaluated_suppliers: list[_EvaluatedSupplier], overall_reasoning: str) -> str:
    """Terminal tool: submit your final supplier evaluation and reasoning. Call this exactly
    once, after you've decided which supplier to recommend. total_score is 0-100. Exactly one
    supplier must have is_recommended=true."""
    db = SupabaseRepository()
    suppliers = [s.model_dump() for s in evaluated_suppliers]
    recommended = next((s for s in suppliers if s["is_recommended"]), None)
    decision_json = {
        "evaluated_suppliers": suppliers,
        "overall_reasoning": overall_reasoning,
        "recommended_supplier_id": recommended["supplier_id"] if recommended else None,
    }
    db.write_audit_log(
        action_type="SUPPLIER_EVALUATION",
        agent_name="evaluation_agent",
        decision_json=decision_json,
    )
    return "Audit log recorded."


_TOOLS = [get_purchase_history, get_reference_score, write_audit_log]

_SYSTEM_PROMPT = (
    "You are the Evaluation specialist for a procurement system. Judge the supplier quotes on "
    "price, delivery time, and payment terms. Call get_purchase_history for historical context "
    "and get_reference_score for a deterministic reference point, but use your own judgment — "
    "flag any risk you notice (e.g. price far above historical average, unusually long "
    "delivery). Call write_audit_log exactly once, when you're done, with your final decision."
)


logger = logging.getLogger(__name__)


def _history_entry(summary: str) -> dict[str, str]:
    return {"worker": "evaluation", "summary": summary}


async def evaluation_node(state: ProcurementState) -> ProcurementState:
    """Run the Evaluation ReAct agent once; it must finish by calling write_audit_log."""
    history = state.get("supervisor_history", [])
    try:
        agent = create_react_agent(_build_llm("smart"), _TOOLS, prompt=_SYSTEM_PROMPT)
        prompt = (
            f"item_id: {state.get('item_id')}\nextracted_quotes: {state.get('extracted_quotes')}\n"
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        args = _last_tool_call(result["messages"], "write_audit_log")
        if args is None:
            raise ValueError("Evaluation agent finished without calling write_audit_log")

        return {
            **state,
            "evaluated_suppliers": args["evaluated_suppliers"],
            "supervisor_history": [*history, _history_entry(args["overall_reasoning"][:120])],
        }
    except Exception as exc:
        logger.exception("evaluation worker failed")
        return {
            **state,
            "error": str(exc),
            "supervisor_history": [*history, _history_entry(f"FAILED: {exc}")],
        }
