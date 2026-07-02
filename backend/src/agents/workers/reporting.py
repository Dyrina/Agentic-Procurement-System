"""agents/workers/reporting.py — Reporting specialist: executive summary + comparison table.

The comparison table (generate_report) is a deterministic template — there's no decision to
make, so it isn't wrapped in a tool-calling loop. The one place an LLM adds real value here is
writing the executive summary, which is a single plain generation call, not a tool-selection
decision, so no create_react_agent is needed either.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agents.tools.report import generate_report
from src.agents.workers import _build_llm, _extract_text
from src.core.state import ProcurementState


def _history_entry(summary: str) -> dict[str, str]:
    return {"worker": "reporting", "summary": summary}


async def _write_executive_summary(state: ProcurementState) -> str:
    recommended = next(
        (s for s in state.get("evaluated_suppliers", []) if s.get("is_recommended")), None
    )
    prompt = (
        "Write a concise executive summary (3-4 sentences) of this procurement decision for a "
        "manager who needs to approve or reject it.\n\n"
        f"Item: {state.get('item_name')}\n"
        f"Requested quantity: {state.get('requested_qty')}\n"
        f"Stock sufficient: {state.get('stock_sufficient')}\n"
        f"Recommended supplier: {recommended}\n"
    )
    response = await _build_llm().ainvoke([HumanMessage(content=prompt)])
    return _extract_text(response.content)


async def reporting_node(state: ProcurementState) -> ProcurementState:
    """Assemble the deterministic comparison table plus an LLM-written executive summary."""
    history = state.get("supervisor_history", [])
    try:
        report_result = await generate_report(
            evaluated_suppliers=state.get("evaluated_suppliers", []),
            item_name=state.get("item_name", ""),
            requested_qty=state.get("requested_qty", 0),
            stock_sufficient=state.get("stock_sufficient"),
            current_stock=state.get("current_stock"),
        )
        summary = await _write_executive_summary(state)
        report_markdown = f"## Executive Summary\n\n{summary}\n\n{report_result['report_markdown']}"
        return {
            **state,
            "report_markdown": report_markdown,
            "supervisor_history": [*history, _history_entry("report assembled")],
        }
    except Exception as exc:
        return {
            **state,
            "error": str(exc),
            "supervisor_history": [*history, _history_entry(f"FAILED: {exc}")],
        }
