"""agents/manager.py — supervisor + worker-agent LangGraph state machine.

Replaces the old single-shot plan_node/validate_node/execute_node pipeline: instead of an LLM
picking a fixed step list once upfront and running it blind, supervisor_node is re-invoked
after every worker returns, so it can skip steps that turned out unnecessary (e.g. stock
already sufficient) or route to "fail" the moment a worker reports an error, instead of a
bare exception aborting the whole session.

Ordering between workers is enforced structurally by data availability (Evaluation has
nothing to call on until Sourcing has produced extracted_quotes) rather than by a separate
plan-validation layer — there is no equivalent of the old validate_plan/ORDERING_CONSTRAINTS.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.agents.workers.evaluation import evaluation_node
from src.agents.workers.intake import intake_await_node, intake_node
from src.agents.workers.inventory import inventory_await_node, inventory_node
from src.agents.workers.reporting import reporting_node
from src.agents.workers.sourcing import sourcing_await_node, sourcing_node
from src.core.config import get_checkpointer, get_settings
from src.core.state import ProcurementState

# ponytail: flat safety cap against a runaway supervisor loop, raise if real flows need more
_MAX_WORKER_CALLS = 15

_WORKER_NAMES = ("intake", "inventory", "sourcing", "evaluation", "reporting")


class _NextWorker(BaseModel):
    next: Literal["intake", "inventory", "sourcing", "evaluation", "reporting", "stop", "fail"]
    reason: str


# ── Supervisor node ──────────────────────────────────────────────────────────


async def supervisor_node(state: ProcurementState) -> ProcurementState:
    """Decide the next worker to run given the current state, or stop/fail."""
    if state.get("error"):
        return {**state, "next_worker": "fail"}
    if state.get("report_markdown"):
        # Deterministic, not left to the LLM's judgment: a real Gemini call re-ran Reporting
        # 4 times before deciding to stop, since it only sees a short text summary of history
        # ("reporting: report assembled"), not the actual report_markdown content, and
        # apparently found that ambiguous. Checking the field directly is free and infallible.
        return {**state, "next_worker": "stop"}

    calls = state.get("worker_calls", 0) + 1
    if calls > _MAX_WORKER_CALLS:
        return {
            **state,
            "next_worker": "fail",
            "error": "Exceeded max supervisor iterations",
            "worker_calls": calls,
        }

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", google_api_key=settings.GOOGLE_API_KEY
    )
    structured = llm.with_structured_output(_NextWorker)

    history_lines = "\n".join(
        f"- {h['worker']}: {h['summary']}" for h in state.get("supervisor_history", [])
    )
    prompt = (
        "You are the procurement supervisor. Pick the next specialist to run, or 'stop' once "
        "report_markdown is ready, or 'fail' if the request genuinely cannot proceed.\n\n"
        f"User request: {state.get('user_message', '')}\n\n"
        f"Progress so far:\n{history_lines or '(nothing yet)'}\n\n"
        "Guidance:\n"
        "- intake first if item_name/requested_qty are not yet known.\n"
        "- inventory next, to resolve the item and check stock.\n"
        "- if stock_sufficient is true, skip sourcing/evaluation and go straight to reporting.\n"
        "- sourcing before evaluation (evaluation needs extracted_quotes).\n"
        "- stop once report_markdown exists."
    )
    result = await structured.ainvoke([HumanMessage(content=prompt)])
    return {**state, "next_worker": result.next, "worker_calls": calls}


def _route_from_supervisor(state: ProcurementState) -> str:
    next_worker = state["next_worker"]
    if next_worker == "stop":
        return "finalize"
    if next_worker == "fail":
        return "error"
    return next_worker


def _route_on_clarification(state: ProcurementState, await_node: str) -> str:
    return await_node if state.get("needs_clarification") else "supervisor"


def _route_after_intake(state: ProcurementState) -> str:
    return _route_on_clarification(state, "intake_await")


def _route_after_inventory(state: ProcurementState) -> str:
    return _route_on_clarification(state, "inventory_await")


def _route_after_sourcing(state: ProcurementState) -> str:
    return _route_on_clarification(state, "sourcing_await")


# ── Terminal nodes ───────────────────────────────────────────────────────────


async def finalize_node(state: ProcurementState) -> ProcurementState:
    return {**state, "status": "AWAITING_APPROVAL"}


async def error_node(state: ProcurementState) -> ProcurementState:
    return {**state, "status": "FAILED"}


# ── Graph construction ───────────────────────────────────────────────────────


def build_manager_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(ProcurementState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("intake", intake_node)
    graph.add_node("intake_await", intake_await_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("inventory_await", inventory_await_node)
    graph.add_node("sourcing", sourcing_node)
    graph.add_node("sourcing_await", sourcing_await_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("reporting", reporting_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("error", error_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {**{name: name for name in _WORKER_NAMES}, "finalize": "finalize", "error": "error"},
    )

    graph.add_conditional_edges(
        "intake", _route_after_intake, {"intake_await": "intake_await", "supervisor": "supervisor"}
    )
    graph.add_edge("intake_await", "intake")

    graph.add_conditional_edges(
        "inventory",
        _route_after_inventory,
        {"inventory_await": "inventory_await", "supervisor": "supervisor"},
    )
    graph.add_edge("inventory_await", "inventory")

    graph.add_conditional_edges(
        "sourcing",
        _route_after_sourcing,
        {"sourcing_await": "sourcing_await", "supervisor": "supervisor"},
    )
    graph.add_edge("sourcing_await", "sourcing")

    graph.add_edge("evaluation", "supervisor")
    graph.add_edge("reporting", "supervisor")
    graph.add_edge("finalize", END)
    graph.add_edge("error", END)

    return graph.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_manager_graph():
    """Build (once) and cache the production graph, backed by the real Postgres checkpointer.

    Must only be called from an async context (e.g. inside a FastAPI request handler or the
    app lifespan) — AsyncPostgresSaver requires a running event loop at construction time, so
    this cannot be a bare module-level statement evaluated at import time.
    """
    return build_manager_graph(checkpointer=get_checkpointer())
