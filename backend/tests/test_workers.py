"""Unit tests for the supervisor-loop worker agents (src/agents/workers/*.py)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.workers import _last_tool_call
from src.core.state import ProcurementState


# ── shared helpers ───────────────────────────────────────────────────────────

def test_last_tool_call_finds_matching_call():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "search_items", "args": {"query": "laptop"}, "id": "1"}]),
    ]
    assert _last_tool_call(messages, "search_items") == {"query": "laptop"}


def test_last_tool_call_returns_most_recent_when_called_twice():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "submit_intake", "args": {"item_name": "old"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "submit_intake", "args": {"item_name": "new"}, "id": "2"}]),
    ]
    assert _last_tool_call(messages, "submit_intake") == {"item_name": "new"}


def test_last_tool_call_returns_none_when_absent():
    messages = [HumanMessage(content="hi"), AIMessage(content="no tools here")]
    assert _last_tool_call(messages, "submit_intake") is None


# ── intake ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intake_node_submits_parsed_request(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm([
        AIMessage(content="", tool_calls=[{
            "name": "submit_intake",
            "args": {"item_name": "Dell XPS 15 Laptop", "requested_qty": 30},
            "id": "1",
        }]),
        AIMessage(content="done"),
    ])
    state: ProcurementState = {"session_id": "s1", "user_message": "order 30 units of Dell XPS 15 laptop"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["item_name"] == "Dell XPS 15 Laptop"
    assert result["requested_qty"] == 30
    assert result["needs_clarification"] is False
    assert "error" not in result


@pytest.mark.asyncio
async def test_intake_node_flags_ambiguous_request(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm([
        AIMessage(content="", tool_calls=[{
            "name": "flag_ambiguous",
            "args": {"question": "How many units do you need?"},
            "id": "1",
        }]),
    ])
    state: ProcurementState = {"session_id": "s1", "user_message": "I need some laptops"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["needs_clarification"] is True
    assert result["clarification_payload"]["question"] == "How many units do you need?"
    assert result["intake_attempts"] == 1


@pytest.mark.asyncio
async def test_intake_node_fails_after_max_ambiguous_attempts(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm([
        AIMessage(content="", tool_calls=[{
            "name": "flag_ambiguous", "args": {"question": "still unclear"}, "id": "1",
        }]),
    ])
    state: ProcurementState = {"session_id": "s1", "user_message": "laptops", "intake_attempts": 2}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["needs_clarification"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_intake_node_catches_exceptions():
    from src.agents.workers.intake import intake_node

    state: ProcurementState = {"session_id": "s1", "user_message": "laptops"}
    with patch("src.agents.workers.intake._build_llm", side_effect=RuntimeError("boom")):
        result = await intake_node(state)

    assert result["error"] == "boom"
    assert "FAILED" in result["supervisor_history"][-1]["summary"]


@pytest.mark.asyncio
async def test_intake_await_node_passthrough_when_not_needed():
    from src.agents.workers.intake import intake_await_node

    state: ProcurementState = {"session_id": "s1", "needs_clarification": False}
    result = await intake_await_node(state)
    assert result == state


@pytest.mark.asyncio
async def test_intake_await_node_resumes_with_clarified_message():
    from src.agents.workers.intake import intake_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "laptops",
        "needs_clarification": True,
        "clarification_payload": {"type": "intake_clarification", "question": "how many?"},
    }
    with patch("src.agents.workers.intake.interrupt", return_value="30 units"):
        result = await intake_await_node(state)

    assert result["needs_clarification"] is False
    assert result["clarification_payload"] is None
    assert "30 units" in result["user_message"]


# ── inventory ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_inventory_node_always_asks_for_confirmation(fake_llm):
    from src.agents.workers.inventory import inventory_node

    candidates = [{"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "similarity": 0.9}]
    llm = fake_llm([
        AIMessage(content="", tool_calls=[{
            "name": "search_items", "args": {"query": "Dell XPS 15"}, "id": "1",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "ask_user_to_confirm",
            "args": {"candidates": candidates, "question": "Did you mean Dell XPS 15 Laptop?"},
            "id": "2",
        }]),
    ])
    mock_db = MagicMock()
    mock_db.rpc.return_value = candidates
    state: ProcurementState = {"session_id": "s1", "item_name": "Dell XPS 15", "requested_qty": 30}
    with patch("src.agents.workers.inventory._build_llm", return_value=llm), \
         patch("src.agents.workers.inventory.SupabaseRepository", return_value=mock_db):
        result = await inventory_node(state)

    assert result["needs_clarification"] is True
    assert result["clarification_payload"]["candidates"] == candidates
    assert "item_id" not in result  # never silently picked


@pytest.mark.asyncio
async def test_inventory_node_fails_if_agent_skips_confirmation(fake_llm):
    from src.agents.workers.inventory import inventory_node

    llm = fake_llm([AIMessage(content="Dell XPS 15 Laptop it is.")])  # no tool call at all
    state: ProcurementState = {"session_id": "s1", "item_name": "Dell XPS 15", "requested_qty": 30}
    with patch("src.agents.workers.inventory._build_llm", return_value=llm):
        result = await inventory_node(state)

    assert "error" in result


@pytest.mark.asyncio
async def test_inventory_node_checks_stock_once_item_id_confirmed():
    from src.agents.workers.inventory import inventory_node

    mock_db = MagicMock()
    mock_db.get_item.return_value = {"item_id": "IT-XPS-15", "current_stock": 4}
    state: ProcurementState = {"session_id": "s1", "item_id": "IT-XPS-15", "requested_qty": 30}
    with patch("src.agents.workers.inventory.SupabaseRepository", return_value=mock_db):
        result = await inventory_node(state)

    assert result["current_stock"] == 4
    assert result["stock_sufficient"] is False


@pytest.mark.asyncio
async def test_inventory_await_node_resumes_with_selected_item_id():
    from src.agents.workers.inventory import inventory_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "needs_clarification": True,
        "clarification_payload": {"type": "inventory_candidate_confirm", "candidates": [], "question": "?"},
    }
    with patch("src.agents.workers.inventory.interrupt", return_value={"selected_item_id": "IT-XPS-15"}):
        result = await inventory_await_node(state)

    assert result["item_id"] == "IT-XPS-15"
    assert result["needs_clarification"] is False
