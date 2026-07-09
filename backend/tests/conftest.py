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
