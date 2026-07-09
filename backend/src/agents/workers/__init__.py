"""agents/workers — shared helpers for the 5 supervisor-loop worker agents."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import get_settings


def _build_llm(tier: str = "fast") -> ChatGoogleGenerativeAI:
    """Construct the Gemini model for a worker. tier="fast" (default) for tasks whose errors
    are caught downstream by a schema or a human gate; tier="smart" for judgment where errors
    cost money or land verbatim in front of the user (quote parsing, evaluation, report prose).
    timeout/max_retries so one hung Gemini call can't freeze a session forever.
    60s not 30 — the smart tier is a thinking model and legitimately runs past 30s."""
    settings = get_settings()
    model = settings.MODEL_SMART if tier == "smart" else settings.MODEL_FAST
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=settings.GOOGLE_API_KEY, timeout=60, max_retries=2
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


def _cancel_requested(answer: Any) -> bool:
    """True when an interrupt() resume payload is the universal cancel action — every await
    node must honour it so the user always has an exit."""
    return isinstance(answer, dict) and answer.get("action") == "cancel"


def _extract_text(content: str | list) -> str:
    """AIMessage.content is usually a plain string, but Gemini 3 'thinking' models return a
    list of content parts (each carrying a thought_signature) instead — pull just the text out."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


def _format_error(exc: Exception) -> str:
    """Format unhandled worker exceptions into clean, user-facing error messages."""
    msg = str(exc)
    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return "Configuration Error: The provided AI Model API Key is invalid or missing."
    if "Resource exhausted" in msg or "Quota exceeded" in msg or "429" in msg:
        return "Service Error: The AI service's rate limit or quota has been exceeded."
    # We fallback to a generic message so ugly JSON tracebacks don't leak to the UI
    return f"An unexpected system error occurred ({exc.__class__.__name__}). Please check the server logs."
"""agents/workers — shared helpers for the 5 supervisor-loop worker agents."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import get_settings


def _build_llm(tier: str = "fast") -> ChatGoogleGenerativeAI:
    """Construct the Gemini model for a worker. tier="fast" (default) for tasks whose errors
    are caught downstream by a schema or a human gate; tier="smart" for judgment where errors
    cost money or land verbatim in front of the user (quote parsing, evaluation, report prose).
    timeout/max_retries so one hung Gemini call can't freeze a session forever.
    60s not 30 — the smart tier is a thinking model and legitimately runs past 30s."""
    settings = get_settings()
    model = settings.MODEL_SMART if tier == "smart" else settings.MODEL_FAST
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=settings.GOOGLE_API_KEY, timeout=60, max_retries=2
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


def _cancel_requested(answer: Any) -> bool:
    """True when an interrupt() resume payload is the universal cancel action — every await
    node must honour it so the user always has an exit."""
    return isinstance(answer, dict) and answer.get("action") == "cancel"


def _extract_text(content: str | list) -> str:
    """AIMessage.content is usually a plain string, but Gemini 3 'thinking' models return a
    list of content parts (each carrying a thought_signature) instead — pull just the text out."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
