"""Shared pytest fixtures for testing create_react_agent-based workers without a real LLM."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class FakeToolCallingModel(FakeMessagesListChatModel):
    """BaseChatModel double for create_react_agent: plays back canned AIMessage responses.

    create_react_agent calls model.bind_tools(tools) once and then drives the returned
    model with .ainvoke(messages) in a loop until a response has no tool_calls. The base
    FakeMessagesListChatModel supports the .ainvoke() playback; bind_tools just needs to
    not raise (the default BaseChatModel.bind_tools raises NotImplementedError).
    """

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def fake_llm():
    """fake_llm([AIMessage(...), ...]) -> a model create_react_agent can run against."""

    def _make(responses):
        return FakeToolCallingModel(responses=list(responses))

    return _make


@pytest.fixture
def fake_supervisor_llm():
    """fake_supervisor_llm([decision, ...]) -> stand-in for supervisor.py's
    ChatGoogleGenerativeAI(...).with_structured_output(_NextWorker) chain. Pops one decision
    per supervisor_node call from a shared queue (not a fresh copy) so a multi-turn graph run
    advances through the queue instead of restarting it every call.
    """

    def _make(decisions):
        shared = list(decisions)

        class _FakeStructured:
            async def ainvoke(self, messages):
                return shared.pop(0)

        class _FakeLLM:
            def with_structured_output(self, schema):
                return _FakeStructured()

        return _FakeLLM()

    return _make
