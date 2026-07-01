from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.agents.tools.evaluation import evaluate_suppliers_handler
from src.agents.tools.history import query_history_handler
from src.agents.tools.quotes import extract_quotes_handler, wait_for_quotes_handler
from src.agents.tools.report import generate_report_handler
from src.agents.tools.rfq import send_rfqs_handler
from src.agents.tools.stock import check_stock_handler
from src.api.sse import end_stream, push_event
from src.core.config import get_settings
from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.mcp_server import TOOL_DESCRIPTIONS, validate_plan

# ── Tool registry: step name → async handler function ─────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "check_stock": check_stock_handler,
    "send_rfqs": send_rfqs_handler,
    "wait_for_quotes": wait_for_quotes_handler,
    "extract_quotes": extract_quotes_handler,
    "query_history": query_history_handler,
    "evaluate_suppliers": evaluate_suppliers_handler,
    "generate_report": generate_report_handler,
}

STEP_MESSAGES: dict[str, str] = {
    "check_stock": "Checking inventory stock levels...",
    "send_rfqs": "Drafting and sending RFQ emails to suppliers...",
    "wait_for_quotes": "Waiting for supplier replies (checking every 15 seconds)...",
    "extract_quotes": "Reading supplier replies and extracting quote data...",
    "query_history": "Querying historical purchase data...",
    "evaluate_suppliers": "Evaluating and scoring suppliers...",
    "generate_report": "Assembling recommendation report...",
}


# ── Pydantic schema for Gemini structured output ───────────────────────────

class _PlanOutput(BaseModel):
    plan: list[str]


# ── Graph nodes ────────────────────────────────────────────────────────────

async def plan_node(state: ProcurementState) -> ProcurementState:
    """Call Gemini with structured output to generate a step-by-step plan."""
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GOOGLE_API_KEY)
    structured = llm.with_structured_output(_PlanOutput)

    tool_list = "\n".join(
        f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
    )
    error_context = ""
    if state.get("plan_error"):
        error_context = f"\n\nPrevious plan was invalid: {state['plan_error']}\nPlease fix the ordering and try again."

    prompt = (
        f"You are a procurement orchestrator. Given the user's request, choose the minimum "
        f"set of steps needed and return them as an ordered plan.\n\n"
        f"Available steps:\n{tool_list}\n\n"
        f"Ordering rules:\n"
        f"- send_rfqs must come before wait_for_quotes\n"
        f"- wait_for_quotes must come before extract_quotes\n"
        f"- extract_quotes must come before evaluate_suppliers\n"
        f"- query_history must come before evaluate_suppliers\n"
        f"- evaluate_suppliers must come before generate_report\n"
        f"{error_context}\n\n"
        f"User request: {state['user_message']}"
    )

    result = await structured.ainvoke([HumanMessage(content=prompt)])
    attempts = state.get("plan_attempts", 0) + 1
    return {**state, "plan": result.plan, "plan_attempts": attempts, "plan_error": None}


async def validate_node(state: ProcurementState) -> ProcurementState:
    """Validate the generated plan against registry + ordering rules."""
    error = validate_plan(state.get("plan", []))
    return {**state, "plan_error": error, "validation_passed": error is None}


def _route_after_validate(state: ProcurementState) -> str:
    if state.get("validation_passed"):
        return "stream_plan"
    if state.get("plan_attempts", 0) < 2:
        return "plan"
    return "error"


async def stream_plan_node(state: ProcurementState) -> ProcurementState:
    """Push the validated plan to the user via SSE."""
    plan = state.get("plan", [])
    readable = " → ".join(step.replace("_", " ") for step in plan)
    await push_event(
        state["session_id"],
        "plan",
        {"steps": plan, "message": f"My plan: {readable}"},
    )
    return {**state, "status": "EXECUTING"}


async def execute_node(state: ProcurementState) -> ProcurementState:
    """Execute each step in the plan, streaming status, persisting state after each step."""
    db = SupabaseRepository()
    session_id = state["session_id"]

    for step in state.get("plan", []):
        await push_event(session_id, "step_start", {"step": step, "message": STEP_MESSAGES.get(step, step)})
        try:
            state = await TOOL_REGISTRY[step](state)
        except Exception as exc:
            error_msg = f"Step '{step}' failed: {exc}"
            await push_event(session_id, "error", {"step": step, "message": error_msg})
            await end_stream(session_id)
            db.update_evaluation(session_id, status="FAILED", current_step=step, state_json=dict(state))
            return {**state, "status": "FAILED", "error": error_msg, "current_step": step}

        state["current_step"] = step
        db.update_evaluation(session_id, current_step=step, state_json=dict(state))
        await push_event(session_id, "step_done", {"step": step, "message": f"✓ Completed"})

    await push_event(session_id, "report", {"markdown": state.get("report_markdown", "")})
    await push_event(session_id, "approve_ready", {"session_id": session_id, "message": "Approve to generate purchase order"})
    await end_stream(session_id)

    state["status"] = "AWAITING_APPROVAL"
    db.update_evaluation(
        session_id,
        status="AWAITING_APPROVAL",
        report_markdown=state.get("report_markdown", ""),
        state_json=dict(state),
    )
    return state


async def error_node(state: ProcurementState) -> ProcurementState:
    """Stream planning error to user and mark session as FAILED."""
    db = SupabaseRepository()
    session_id = state["session_id"]
    error_msg = state.get("plan_error", "Failed to generate a valid plan after 2 attempts")
    await push_event(session_id, "error", {"step": "planning", "message": error_msg})
    await end_stream(session_id)
    db.update_evaluation(session_id, status="FAILED")
    return {**state, "status": "FAILED"}


# ── Graph construction ──────────────────────────────────────────────────────

def build_manager_graph():
    graph = StateGraph(ProcurementState)

    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("stream_plan", stream_plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("error", error_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"stream_plan": "stream_plan", "plan": "plan", "error": "error"},
    )
    graph.add_edge("stream_plan", "execute")
    graph.add_edge("execute", END)
    graph.add_edge("error", END)

    return graph.compile()


manager_graph = build_manager_graph()
