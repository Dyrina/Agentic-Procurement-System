"""agents/workers/inventory.py — Inventory specialist: resolves the item and checks stock.

Replaces agents/tools/stock.py's exact-match-then-substring-containment logic (which took
rows[0] blindly and raised ValueError on any miss) with pg_trgm similarity search plus a
mandatory human confirmation step — the agent can never silently pick a candidate, even a
single clear match, because there is no tool that lets it finish without asking first.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from src.agents.workers import _build_llm, _cancel_requested, _format_error, _last_tool_call
from src.core.state import ProcurementState
from src.database.client import SupabaseRepository


@tool
def search_items(query: str) -> list[dict]:
    """Search the item catalog by approximate name (trigram similarity); up to 5 candidates.
    Returns an empty list if nothing matches — that's a valid outcome, not an error."""
    db = SupabaseRepository()
    return db.rpc("search_items_by_name", {"query": query, "match_limit": 5})


@tool
def ask_user_to_confirm(candidates: list[dict], question: str) -> str:
    """Terminal tool: present candidate items and ask the user to pick one. Call this every
    time after search_items — never proceed without asking, even if only one candidate came back."""
    return "Awaiting user confirmation."


_TOOLS = [search_items, ask_user_to_confirm]

# Always offered alongside real candidates — pg_trgm's `%` similarity match can return weak,
# irrelevant hits for almost any query (short/common words especially), so "zero candidates"
# isn't the only way a genuinely new item shows up here. Selecting this resolves to the same
# UNCATALOGED sentinel the zero-match pre-check below uses.
_NOT_LISTED_OPTION = {
    "item_id": "UNCATALOGED",
    "name": "None of these — it's a new item",
    "category": "",
    "current_stock": 0,
    "similarity": 0,
}

_SYSTEM_PROMPT = (
    "You are the Inventory specialist for a procurement system. The user's request involves "
    "an item described as {item_name!r}. Call search_items with a good search query to find "
    "candidate items in the catalog, then ALWAYS call ask_user_to_confirm with the candidates "
    "and a short question — even if there is only one clear match, you must never pick an item "
    "without asking the user first. This is a hard requirement, not a suggestion."
)

# The confirmation question shown to the human is deterministic, not the LLM's — the wording
# must match the intent ("would you like to purchase?" on a stock check is wrong), and that is
# too important to leave to model mood. The agent still authors a question via
# ask_user_to_confirm; it just isn't the one displayed.
_CONFIRM_QUESTION_CHECK = "I found these in the catalog — which one should I check stock for?"
_CONFIRM_QUESTION_DEFAULT = "I found these in the catalog — which one is the item you need?"


logger = logging.getLogger(__name__)


def _history_entry(summary: str) -> dict[str, str]:
    return {"worker": "inventory", "summary": summary}


async def _check_stock(item_id: str, requested_qty: int) -> dict:
    """Deterministic stock lookup for an already-resolved item_id — no LLM needed here."""
    db = SupabaseRepository()
    item = await asyncio.to_thread(db.get_item, item_id)
    if item is None:
        raise ValueError(f"Item '{item_id}' not found")
    return {
        "current_stock": item["current_stock"],
        "stock_sufficient": item["current_stock"] >= requested_qty,
        "item_category": item["category"],
    }


async def inventory_node(state: ProcurementState) -> ProcurementState:
    """Resolve item_id (asking the user to confirm candidates) then check stock."""
    history = state.get("supervisor_history", [])
    try:
        if state.get("item_id"):
            # Already confirmed by the user in a previous round — just check stock.
            result = await _check_stock(state["item_id"], state["requested_qty"])
            return {
                **state,
                "current_stock": result["current_stock"],
                "stock_sufficient": result["stock_sufficient"],
                "item_category": result["item_category"],
                "inventory_candidates": None,
                "supervisor_history": [
                    *history,
                    _history_entry(
                        f"item_id={state['item_id']!r}, "
                        f"stock_sufficient={result['stock_sufficient']}"
                    ),
                ],
            }

        # Deterministic pre-check, same reasoning as the item_id branch above: no need for an
        # LLM round-trip just to discover the catalog has zero matches. Buying something not
        # yet in the catalog is a normal, common procurement flow (a non-stock/one-time
        # purchase) — not an error — so this proceeds straight to sourcing under a shared
        # sentinel item_id rather than interrupting to ask, and rather than auto-creating a
        # real catalog row (that's someone else's system's job, not this agent's).
        db = SupabaseRepository()
        candidates = await asyncio.to_thread(
            db.rpc, "search_items_by_name", {"query": state["item_name"], "match_limit": 5}
        )
        if not candidates:
            return {
                **state,
                "item_id": "UNCATALOGED",
                # Not in the catalog = zero on hand, by definition — no DB lookup needed.
                "current_stock": 0,
                "stock_sufficient": False,
                "item_category": "Uncataloged",
                "inventory_candidates": None,
                "supervisor_history": [
                    *history,
                    _history_entry(
                        f"{state['item_name']!r} not in catalog — proceeding as a non-stock item"
                    ),
                ],
            }

        agent = create_react_agent(
            _build_llm(), _TOOLS, prompt=_SYSTEM_PROMPT.format(item_name=state["item_name"])
        )
        message = f"Item requested: {state['item_name']} (qty {state['requested_qty']})"
        result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
        args = _last_tool_call(result["messages"], "ask_user_to_confirm")
        if args is None:
            raise ValueError(
                "Inventory agent finished without asking the user to confirm a candidate"
            )

        candidates = [*args["candidates"], _NOT_LISTED_OPTION]
        question = (
            _CONFIRM_QUESTION_CHECK
            if state.get("intent") == "check_stock"
            else _CONFIRM_QUESTION_DEFAULT
        )
        return {
            **state,
            "needs_clarification": True,
            "inventory_candidates": candidates,
            "clarification_payload": {
                "type": "inventory_candidate_confirm",
                "question": question,
                "candidates": candidates,
            },
        }
    except Exception as exc:
        logger.exception("inventory worker failed")
        return {
            **state,
            "error": _format_error(exc),
            "supervisor_history": [*history, _history_entry(f"FAILED: {exc}")],
        }


async def inventory_await_node(state: ProcurementState) -> ProcurementState:
    """Pause node: interrupt()s with the candidate list, resumes with the selected_item_id."""
    if not state.get("needs_clarification"):
        return state
    answer = interrupt(state["clarification_payload"])
    base = {
        **state,
        "needs_clarification": False,
        "clarification_payload": None,
        "inventory_candidates": None,
    }
    if _cancel_requested(answer):
        return {**base, "cancelled": True}
    return {**base, "item_id": answer["selected_item_id"]}
