"""tests/test_interrupt_resume.py — proves the real LangGraph interrupt()/Command(resume=...)
contract actually suspends and resumes the graph, on top of the per-node unit tests in
test_workers.py (which patch interrupt() directly and only check each await-node's own
read/write contract, not that suspension/resumption genuinely works end to end).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.manager import build_manager_graph


@pytest.mark.asyncio
async def test_graph_suspends_on_clarification_and_resumes_with_answer(fake_llm):
    # Each phase gets its own model instance — a shared FakeMessagesListChatModel cycles its
    # responses forever (it never raises when exhausted), so reusing one across two separate
    # intake_node() invocations would leak the second phase's response into the first.
    ambiguous_phase = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "flag_ambiguous", "args": {"question": "which laptop?"}, "id": "1"}
                ],
            ),
            AIMessage(content="waiting for clarification"),
        ]
    )
    resolved_phase = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {
                            "item_name": "Dell XPS 15",
                            "requested_qty": 1,
                            "intent": "check_stock",
                        },
                        "id": "2",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    reporting_llm = MagicMock()
    reporting_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Summary."))

    with (
        patch(
            "src.agents.workers.intake._build_llm", side_effect=[ambiguous_phase, resolved_phase]
        ),
        patch("src.agents.workers.reporting._build_llm", return_value=reporting_llm),
    ):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t-interrupt"}}

        first = await graph.ainvoke(
            {
                "session_id": "s1",
                "user_id": "u1",
                "user_message": "check the laptop stock",
                "worker_calls": 0,
                # Item already resolved so the resumed run goes intake → reporting without
                # needing a live inventory lookup — this test is about the pause contract.
                "item_id": "IT-XPS-15",
                "current_stock": 4,
                "stock_sufficient": True,
            },
            config=config,
        )
        assert "__interrupt__" in first
        assert first["__interrupt__"][0].value == {
            "type": "intake_clarification",
            "question": "which laptop?",
        }
        # The graph genuinely paused: it never got past Intake to increment worker_calls
        # for a second supervisor turn.
        assert first["worker_calls"] == 1

        resumed = await graph.ainvoke(Command(resume={"text": "Dell XPS 15"}), config=config)

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "COMPLETED"
    assert resumed["item_name"] == "Dell XPS 15"
    assert resumed["requested_qty"] == 1
    assert "(User clarified: Dell XPS 15)" in resumed["user_message"]


@pytest.mark.asyncio
async def test_cancel_at_clarification_gate_ends_session_cancelled(fake_llm):
    intake_llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "flag_ambiguous", "args": {"question": "which laptop?"}, "id": "1"}
                ],
            ),
            AIMessage(content="waiting"),
        ]
    )
    with patch("src.agents.workers.intake._build_llm", return_value=intake_llm):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t-cancel"}}

        first = await graph.ainvoke(
            {
                "session_id": "s3",
                "user_id": "u1",
                "user_message": "buy a laptop",
                "worker_calls": 0,
            },
            config=config,
        )
        assert "__interrupt__" in first

        resumed = await graph.ainvoke(Command(resume={"action": "cancel"}), config=config)

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "CANCELLED"
    assert resumed["cancelled"] is True


@pytest.mark.asyncio
async def test_resuming_without_a_checkpointer_thread_starts_fresh():
    """Sanity check on the mechanism itself: a different thread_id has no saved interrupt to
    resume, so it's a completely independent run — proves state isolation between sessions."""
    graph = build_manager_graph(checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {
            "session_id": "s2",
            "user_id": "u1",
            "user_message": "hello",
            "worker_calls": 0,
            "report_markdown": "## Already done",
        },
        config={"configurable": {"thread_id": "other-thread"}},
    )

    assert "__interrupt__" not in result
    assert result["status"] == "COMPLETED"
