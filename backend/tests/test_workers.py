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
