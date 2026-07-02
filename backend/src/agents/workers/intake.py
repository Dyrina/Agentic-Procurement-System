"""agents/workers/intake.py — Intake specialist: turns free text into item_name/requested_qty.

Replaces routes.py's old _extract_item_name/_extract_quantity whitespace-splitting heuristics.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from src.agents.workers import _build_llm, _last_tool_call
from src.core.state import ProcurementState

_MAX_INTAKE_ATTEMPTS = 3


@tool
def submit_intake(item_name: str, requested_qty: int, constraints: str = "") -> str:
    """Submit the parsed procurement request once the item name and quantity are both clear."""
    return f"Intake recorded: {requested_qty} x {item_name!r}"


@tool
def flag_ambiguous(question: str) -> str:
    """Call instead of submit_intake when the item or quantity is unclear — ask the user."""
    return f"Flagged for clarification: {question}"


_TOOLS = [submit_intake, flag_ambiguous]

_SYSTEM_PROMPT = (
    "You are the Intake specialist for a procurement system. Read the user's request and "
    "extract exactly what they want to buy: item_name (a clean product name, no quantity "
    "words like 'units of' or numbers) and requested_qty (an integer, default 1 if not stated). "
    "If the request is genuinely ambiguous (no identifiable item, or contradictory quantity), "
    "call flag_ambiguous with a specific question instead. Always call exactly one of "
    "submit_intake or flag_ambiguous."
)


def _history_entry(summary: str) -> dict[str, str]:
    return {"worker": "intake", "summary": summary}


async def intake_node(state: ProcurementState) -> ProcurementState:
    """Run the Intake ReAct agent once; write item_name/requested_qty or a clarification request."""
    history = state.get("supervisor_history", [])
    try:
        agent = create_react_agent(_build_llm(), _TOOLS, prompt=_SYSTEM_PROMPT)
        result = await agent.ainvoke({"messages": [HumanMessage(content=state["user_message"])]})
        messages = result["messages"]

        if args := _last_tool_call(messages, "flag_ambiguous"):
            attempts = state.get("intake_attempts", 0) + 1
            if attempts >= _MAX_INTAKE_ATTEMPTS:
                return {
                    **state,
                    "needs_clarification": False,
                    "intake_attempts": attempts,
                    "error": f"Unresolved after {attempts} attempts: {args['question']}",
                    "supervisor_history": [
                        *history,
                        _history_entry("FAILED: too many ambiguous attempts"),
                    ],
                }
            return {
                **state,
                "needs_clarification": True,
                "intake_attempts": attempts,
                "clarification_payload": {
                    "type": "intake_clarification",
                    "question": args["question"],
                },
            }

        args = _last_tool_call(messages, "submit_intake")
        if args is None:
            raise ValueError(
                "Intake agent finished without calling submit_intake or flag_ambiguous"
            )

        return {
            **state,
            "needs_clarification": False,
            "item_name": args["item_name"],
            "requested_qty": int(args["requested_qty"]),
            "supervisor_history": [
                *history,
                _history_entry(f"item_name={args['item_name']!r}, qty={args['requested_qty']}"),
            ],
        }
    except Exception as exc:
        return {
            **state,
            "error": str(exc),
            "supervisor_history": [*history, _history_entry(f"FAILED: {exc}")],
        }


async def intake_await_node(state: ProcurementState) -> ProcurementState:
    """Pause node: interrupt()s with the clarification question, resumes with the user's answer."""
    if not state.get("needs_clarification"):
        return state
    answer = interrupt(state["clarification_payload"])
    return {
        **state,
        "user_message": f"{state['user_message']}\n\n(User clarified: {answer})",
        "needs_clarification": False,
        "clarification_payload": None,
    }
