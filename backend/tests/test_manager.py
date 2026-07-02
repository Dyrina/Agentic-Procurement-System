"""tests/test_manager.py — supervisor routing logic + end-to-end reactive-loop behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.manager import _route_from_supervisor, build_manager_graph, supervisor_node

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
async def test_supervisor_node_short_circuits_to_fail_without_calling_llm():
    """A worker-reported error should route straight to fail — no need to ask the LLM."""

    def _boom(*args, **kwargs):
        raise AssertionError("supervisor should not call the LLM when state already has an error")

    state = {"session_id": "s1", "user_message": "x", "error": "sourcing blew up"}
    with patch("src.agents.manager.ChatGoogleGenerativeAI", side_effect=_boom):
        result = await supervisor_node(state)

    assert result["next_worker"] == "fail"


@pytest.mark.asyncio
async def test_supervisor_node_fails_after_max_worker_calls():
    def _boom(*args, **kwargs):
        raise AssertionError("supervisor should not call the LLM once the cap is exceeded")

    state = {"session_id": "s1", "user_message": "x", "worker_calls": 15}
    with patch("src.agents.manager.ChatGoogleGenerativeAI", side_effect=_boom):
        result = await supervisor_node(state)

    assert result["next_worker"] == "fail"
    assert "error" in result


@pytest.mark.asyncio
async def test_supervisor_node_short_circuits_to_stop_once_report_exists():
    """Regression: a live run against the real Gemini API re-ran Reporting 4 times before
    deciding to stop, because supervisor_history only carries a short text summary
    ("reporting: report assembled"), not the actual report_markdown — apparently ambiguous
    enough for the model to loop. Checking the field directly is deterministic and free."""

    def _boom(*args, **kwargs):
        raise AssertionError("supervisor should not call the LLM once report_markdown exists")

    state = {"session_id": "s1", "user_message": "x", "report_markdown": "## Report\n..."}
    with patch("src.agents.manager.ChatGoogleGenerativeAI", side_effect=_boom):
        result = await supervisor_node(state)

    assert result["next_worker"] == "stop"


@pytest.mark.asyncio
async def test_supervisor_node_asks_llm_and_increments_worker_calls(fake_supervisor_llm):
    llm = fake_supervisor_llm([SimpleNamespace(next="intake", reason="need item info")])
    state = {"session_id": "s1", "user_message": "buy some chairs", "worker_calls": 2}
    with patch("src.agents.manager.ChatGoogleGenerativeAI", return_value=llm):
        result = await supervisor_node(state)

    assert result["next_worker"] == "intake"
    assert result["worker_calls"] == 3


# ── End-to-end reactive loop ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_skips_sourcing_and_evaluation_when_stock_sufficient(
    fake_supervisor_llm, fake_llm
):
    """The flagship behavior this rewrite exists for: the supervisor re-plans after seeing
    Inventory's result and skips Sourcing/Evaluation entirely, instead of running a stale
    upfront plan regardless of what actually happened."""
    supervisor_llm = fake_supervisor_llm(
        [
            SimpleNamespace(next="intake", reason=""),
            SimpleNamespace(next="inventory", reason=""),
            SimpleNamespace(next="reporting", reason=""),
            SimpleNamespace(next="stop", reason=""),
        ]
    )
    intake_llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {"item_name": "Ergonomic Office Chair", "requested_qty": 5},
                        "id": "1",
                    }
                ],
            )
        ]
    )
    inventory_db = MagicMock()
    inventory_db.get_item.return_value = {"item_id": "OF-CHAIR-E", "current_stock": 30}
    reporting_llm = MagicMock()
    reporting_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Stock is sufficient."))

    with (
        patch("src.agents.manager.ChatGoogleGenerativeAI", return_value=supervisor_llm),
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

    assert result["status"] == "AWAITING_APPROVAL"
    assert "report_markdown" in result
    assert result.get("evaluated_suppliers") is None
    assert [h["worker"] for h in result["supervisor_history"]] == [
        "intake",
        "inventory",
        "reporting",
    ]


@pytest.mark.asyncio
async def test_pipeline_stops_with_failed_status_on_worker_error(fake_supervisor_llm):
    supervisor_llm = fake_supervisor_llm([SimpleNamespace(next="intake", reason="")])
    state = {
        "session_id": "s1",
        "user_id": "u1",
        "user_message": "buy something",
        "worker_calls": 0,
    }
    with (
        patch("src.agents.manager.ChatGoogleGenerativeAI", return_value=supervisor_llm),
        patch("src.agents.workers.intake._build_llm", side_effect=RuntimeError("gemini down")),
    ):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "t2"}})

    assert result["status"] == "FAILED"
    assert result["error"] == "gemini down"
