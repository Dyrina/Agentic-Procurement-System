"""agents/workers — shared helpers for the 5 supervisor-loop worker agents."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import get_settings


def _build_llm() -> ChatGoogleGenerativeAI:
    """Construct the small Gemini model used by every worker's ReAct loop."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", google_api_key=get_settings().GOOGLE_API_KEY
    )


def _last_tool_call(messages: list[BaseMessage], tool_name: str) -> dict[str, Any] | None:
    """Return the args of the most recent call to `tool_name`, or None if never called."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            if call["name"] == tool_name:
                return call["args"]
    return None


def _extract_text(content: str | list) -> str:
    """AIMessage.content is usually a plain string, but Gemini 3 'thinking' models return a
    list of content parts (each carrying a thought_signature) instead — pull just the text out."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
