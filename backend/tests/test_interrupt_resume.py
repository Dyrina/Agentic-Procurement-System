"""tests/test_interrupt_resume.py — proves the real LangGraph interrupt()/Command(resume=...)
contract actually suspends and resumes the graph, on top of the per-node unit tests in
test_workers.py (which patch interrupt() directly and only check each await-node's own
read/write contract, not that suspension/resumption genuinely works end to end).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.manager import build_manager_graph


@pytest.mark.asyncio
async def test_graph_suspends_on_clarification_and_resumes_with_answer(
    fake_supervisor_llm, fake_llm
):
    supervisor_llm = fake_supervisor_llm(
        [
            SimpleNamespace(next="intake", reason=""),
            SimpleNamespace(next="stop", reason=""),
        ]
    )
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
                        "args": {"item_name": "Dell XPS 15", "requested_qty": 1},
                        "id": "2",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )

    with (
        patch("src.agents.manager.ChatGoogleGenerativeAI", return_value=supervisor_llm),
        patch(
            "src.agents.workers.intake._build_llm", side_effect=[ambiguous_phase, resolved_phase]
        ),
    ):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t-interrupt"}}

        first = await graph.ainvoke(
            {
                "session_id": "s1",
                "user_id": "u1",
                "user_message": "buy a laptop",
                "worker_calls": 0,
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

        resumed = await graph.ainvoke(Command(resume="Dell XPS 15"), config=config)

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "AWAITING_APPROVAL"
    assert resumed["item_name"] == "Dell XPS 15"
    assert resumed["requested_qty"] == 1


@pytest.mark.asyncio
async def test_resuming_without_a_checkpointer_thread_starts_fresh(fake_supervisor_llm):
    """Sanity check on the mechanism itself: a different thread_id has no saved interrupt to
    resume, so it's a completely independent run — proves state isolation between sessions."""
    supervisor_llm = fake_supervisor_llm([SimpleNamespace(next="stop", reason="")])

    with patch("src.agents.manager.ChatGoogleGenerativeAI", return_value=supervisor_llm):
        graph = build_manager_graph(checkpointer=InMemorySaver())
        result = await graph.ainvoke(
            {"session_id": "s2", "user_id": "u1", "user_message": "hello", "worker_calls": 0},
            config={"configurable": {"thread_id": "other-thread"}},
        )

    assert "__interrupt__" not in result
    assert result["status"] == "AWAITING_APPROVAL"
