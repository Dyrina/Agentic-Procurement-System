"""tests/test_manager.py — deterministic router logic + end-to-end reactive-loop behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.manager import (
    _route_from_supervisor,
    build_manager_graph,
    decide_next,
    supervisor_node,
)

# ── decide_next (pure function, no mocking) ───────────────────────────────────


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        # Terminal conditions take precedence, in order.
        ({"cancelled": True, "error": "boom"}, "stop"),
        ({"error": "sourcing blew up"}, "fail"),
        ({"completion_message": "Out of scope."}, "stop"),
        ({"report_markdown": "## Report"}, "stop"),
        # Pipeline stages by data availability.
        ({"user_message": "buy laptops"}, "intake"),
        ({"item_name": "Laptop", "requested_qty": 1}, "inventory"),
        # Item pre-resolved but stock never checked — inventory still owes the stock check.
        ({"item_name": "Laptop", "item_id": "IT-1", "intent": "buy"}, "inventory"),
        # buy: explicit purchase always sources — stock level is advisory only.
        (
            {"item_name": "L", "item_id": "IT-1", "intent": "buy", "stock_sufficient": True},
            "sourcing",
        ),
        # ensure_stock: source only when stock can't cover the request.
        (
            {"item_name": "L", "item_id": "IT-1", "intent": "ensure_stock", "stock_sufficient": True},
            "reporting",
        ),
        (
            {"item_name": "L", "item_id": "IT-1", "intent": "ensure_stock", "stock_sufficient": False},
            "sourcing",
        ),
        # check_stock: pure query — never reaches sourcing, even when stock is short.
        (
            {"item_name": "L", "item_id": "IT-1", "intent": "check_stock", "stock_sufficient": False},
            "reporting",
        ),
        (
            {
                "item_name": "L",
                "item_id": "IT-1",
                "intent": "buy",
                "stock_sufficient": False,
                "extracted_quotes": [{"supplier_id": "SUP-A"}],
            },
            "evaluation",
        ),
        # Present-but-empty evaluated_suppliers means Evaluation ran — don't loop it.
        (
            {
                "item_name": "L",
                "item_id": "IT-1",
                "intent": "buy",
                "stock_sufficient": False,
                "extracted_quotes": [{"supplier_id": "SUP-A"}],
                "evaluated_suppliers": [],
            },
            "reporting",
        ),
    ],
)
def test_decide_next_routing_table(state, expected):
    assert decide_next(state) == expected


# ── _route_from_supervisor (pure function, no mocking) ────────────────────────


def test_route_from_supervisor_stop_maps_to_finalize():
    assert _route_from_supervisor({"next_worker": "stop"}) == "finalize"


def test_route_from_supervisor_fail_maps_to_error():
    assert _route_from_supervisor({"next_worker": "fail"}) == "error"


@pytest.mark.parametrize("worker", ["intake", "inventory", "sourcing", "evaluation", "reporting"])
def test_route_from_supervisor_worker_names_pass_through(worker):
    assert _route_from_supervisor({"next_worker": worker}) == worker


# ── supervisor_node ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_node_fails_after_max_worker_calls():
    state = {"session_id": "s1", "user_message": "x", "worker_calls": 15}
    result = await supervisor_node(state)

    assert result["next_worker"] == "fail"
    assert "error" in result


@pytest.mark.asyncio
async def test_supervisor_node_increments_worker_calls_and_emits_progress():
    state = {"session_id": "s1", "user_message": "buy some chairs", "worker_calls": 2}
    with patch("src.agents.manager.push_event", new=AsyncMock()) as mock_push:
        result = await supervisor_node(state)

    assert result["next_worker"] == "intake"
    assert result["worker_calls"] == 3
    assert mock_push.call_args.args[1] == "progress"


@pytest.mark.asyncio
async def test_supervisor_node_terminal_routes_skip_cap_and_progress():
    """stop/fail routing must not count against the worker-call cap or emit progress."""
    state = {"session_id": "s1", "report_markdown": "## Report", "worker_calls": 15}
    with patch("src.agents.manager.push_event", new=AsyncMock()) as mock_push:
        result = await supervisor_node(state)

    assert result["next_worker"] == "stop"
    assert result["worker_calls"] == 15
    mock_push.assert_not_awaited()


# ── End-to-end reactive loop ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_skips_sourcing_and_evaluation_for_satisfied_ensure_stock(fake_llm):
    """The flagship routing behavior: an ensure_stock request whose stock already covers the
    need routes inventory → reporting, never touching Sourcing (no RFQ emails go out)."""
    intake_llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {
                            "item_name": "Ergonomic Office Chair",
                            "requested_qty": 5,
                            "intent": "ensure_stock",
                        },
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    inventory_db = MagicMock()
    inventory_db.get_item.return_value = {"item_id": "OF-CHAIR-E", "current_stock": 30}
    reporting_llm = MagicMock()
    reporting_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Stock is sufficient."))

    with (
        patch("src.agents.workers.intake._build_llm", return_value=intake_llm),
        patch("src.agents.workers.inventory.SupabaseRepository", return_value=inventory_db),
        patch("src.agents.workers.reporting._build_llm", return_value=reporting_llm),
    ):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        result = await graph.ainvoke(
            {
                "session_id": "s1",
                "user_id": "u1",
                "user_message": "need 5 office chairs",
                "worker_calls": 0,
                "item_id": "OF-CHAIR-E",
                "requested_qty": 5,
            },
            config={"configurable": {"thread_id": "t1"}},
        )

    assert result["status"] == "COMPLETED"
    assert "report_markdown" in result
    assert result.get("evaluated_suppliers") is None
    assert [h["worker"] for h in result["supervisor_history"]] == [
        "intake",
        "inventory",
        "reporting",
    ]


@pytest.mark.asyncio
async def test_pipeline_completes_with_rejection_for_out_of_scope_request(fake_llm):
    intake_llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reject_out_of_scope",
                        "args": {"reason": "I can only help with procurement requests."},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="rejected"),
        ]
    )
    with patch("src.agents.workers.intake._build_llm", return_value=intake_llm):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        result = await graph.ainvoke(
            {
                "session_id": "s1",
                "user_id": "u1",
                "user_message": "write me a poem",
                "worker_calls": 0,
            },
            config={"configurable": {"thread_id": "t3"}},
        )

    assert result["status"] == "COMPLETED"
    assert result["completion_message"] == "I can only help with procurement requests."
    assert "report_markdown" not in result


@pytest.mark.asyncio
async def test_pipeline_stops_with_failed_status_on_worker_error():
    state = {
        "session_id": "s1",
        "user_id": "u1",
        "user_message": "buy something",
        "worker_calls": 0,
    }
    with patch("src.agents.workers.intake._build_llm", side_effect=RuntimeError("gemini down")):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "t2"}})

    assert result["status"] == "FAILED"
    assert result["error"] == "gemini down"
