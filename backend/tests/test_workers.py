"""Unit tests for the supervisor-loop worker agents (src/agents/workers/*.py)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.workers import _last_tool_call


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
