"""agents/manager.py — supervisor + worker-agent LangGraph state machine.

Layered ownership ("deterministic hub, agentic spokes, human gates"):
  - decide_next (pure code) owns control flow — the procurement pipeline is predictable,
    so routing is derived from data availability, not asked of an LLM.
  - Workers own judgment (intent parsing, quote extraction, supplier evaluation) via their
    own LLM tool-calling loops.
  - Humans own authority via interrupt() gates: item confirmation, sourcing-timeout
    escalation, and the final PO approval.

The supervisor is re-invoked after every worker returns, so it reacts to what actually
happened (skip sourcing on a satisfied ensure_stock request, route to "fail" the moment a
worker reports an error) instead of running a stale upfront plan. Ordering between workers
is enforced structurally by data availability — Evaluation has nothing to work with until
Sourcing has produced extracted_quotes.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from src.agents.workers.evaluation import evaluation_node
from src.agents.workers.intake import intake_await_node, intake_node
from src.agents.workers.inventory import inventory_await_node, inventory_node
from src.agents.workers.reporting import reporting_node
from src.agents.workers.sourcing import sourcing_await_node, sourcing_node
from src.api.sse import push_event
from src.core.config import get_checkpointer
from src.core.state import ProcurementState

# ponytail: flat safety cap against a runaway loop (e.g. sourcing that never yields quotes),
# raise if real flows need more
_MAX_WORKER_CALLS = 15

_WORKER_NAMES = ("intake", "inventory", "sourcing", "evaluation", "reporting")

_PROGRESS_MESSAGES = {
    "intake": "Reading your request...",
    "inventory": "Checking stock levels...",
    "sourcing": "Sourcing quotes from suppliers...",
    "evaluation": "Evaluating supplier quotes...",
    "reporting": "Generating report...",
}


# ── Supervisor: deterministic router ─────────────────────────────────────────


def decide_next(state: ProcurementState) -> str:
    """Pick the next node from state alone.

    Each state field is produced by exactly one worker, so "what's missing" fully determines
    "what runs next". LLMs decide *content* inside the workers (what the user meant, what a
    quote says, which supplier wins); this function decides *sequence*.

    Intent (classified by Intake) gates side effects:
      - "buy": explicit purchase — always source, regardless of stock level.
      - "ensure_stock": top-up — source only if stock can't cover the request.
      - "check_stock": pure query — must NEVER reach sourcing (no outbound emails).
    """
    if state.get("cancelled"):
        return "stop"
    if state.get("error"):
        return "fail"
    if state.get("completion_message"):
        # Intake rejected the request as out-of-scope — the message is the final answer.
        return "stop"
    if state.get("report_markdown"):
        return "stop"
    if not state.get("item_name"):
        return "intake"
    if not state.get("item_id") or "stock_sufficient" not in state:
        # Inventory has two jobs — resolve the item AND check its stock — so it isn't done
        # until both fields exist (item_id can arrive pre-resolved, e.g. after a resume).
        return "inventory"
    if state.get("intent") == "check_stock":
        return "reporting"
    if state.get("intent") == "ensure_stock" and state.get("stock_sufficient"):
        return "reporting"
    if not state.get("extracted_quotes"):
        return "sourcing"
    if "evaluated_suppliers" not in state:
        # Present-but-empty means Evaluation ran and found nothing — don't re-run it.
        return "evaluation"
    return "reporting"


async def supervisor_node(state: ProcurementState) -> ProcurementState:
    """Route to the next worker (with iteration cap + progress event) or stop/fail."""
    next_worker = decide_next(state)
    if next_worker not in _WORKER_NAMES:
        return {**state, "next_worker": next_worker}

    calls = state.get("worker_calls", 0) + 1
    if calls > _MAX_WORKER_CALLS:
        return {
            **state,
            "next_worker": "fail",
            "error": "Exceeded max supervisor iterations",
            "worker_calls": calls,
        }
    await push_event(
        state["session_id"],
        "progress",
        {"step": next_worker, "message": _PROGRESS_MESSAGES[next_worker]},
    )
    return {**state, "next_worker": next_worker, "worker_calls": calls}


def _route_from_supervisor(state: ProcurementState) -> str:
    next_worker = state["next_worker"]
    if next_worker == "stop":
        return "finalize"
    if next_worker == "fail":
        return "error"
    return next_worker


def _route_after_worker(await_node: str):
    """Worker → its await node when clarification is pending, else back to the supervisor."""

    def route(state: ProcurementState) -> str:
        return await_node if state.get("needs_clarification") else "supervisor"

    return route


def _route_after_await(worker: str):
    """Await → back to its worker with the answer, or to the supervisor if the user cancelled."""

    def route(state: ProcurementState) -> str:
        return "supervisor" if state.get("cancelled") else worker

    return route


# ── Terminal nodes ───────────────────────────────────────────────────────────


async def finalize_node(state: ProcurementState) -> ProcurementState:
    if state.get("cancelled"):
        return {**state, "status": "CANCELLED"}
    if state.get("evaluated_suppliers"):
        return {**state, "status": "AWAITING_APPROVAL"}
    # check_stock queries, satisfied ensure_stock requests, out-of-scope rejections.
    return {**state, "status": "COMPLETED"}


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

    for worker in ("intake", "inventory", "sourcing"):
        await_node = f"{worker}_await"
        graph.add_conditional_edges(
            worker,
            _route_after_worker(await_node),
            {await_node: await_node, "supervisor": "supervisor"},
        )
        graph.add_conditional_edges(
            await_node,
            _route_after_await(worker),
            {worker: worker, "supervisor": "supervisor"},
        )

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
