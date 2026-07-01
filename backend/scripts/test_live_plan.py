"""
Manual live test for LLM orchestration (plan -> validate -> stream_plan).

Calls real Gemini via plan_node. Skips execute_node/error_node so no
Supabase or Gmail calls happen. Requires GOOGLE_API_KEY with quota.

Usage:
    cd backend && uv run python -m scripts.test_live_plan "Buy 30 Dell XPS 15 laptops"
"""

from __future__ import annotations

import asyncio
import sys

from src.agents.manager import _route_after_validate, plan_node, stream_plan_node, validate_node
from src.api.sse import create_session, get_queue


async def main(user_message: str) -> None:
    session_id = "live_test_1"
    create_session(session_id)
    state = {"session_id": session_id, "user_message": user_message, "plan_attempts": 0}

    state = await plan_node(state)
    print("PLAN:", state["plan"])

    state = await validate_node(state)
    print("VALID:", state["validation_passed"], "| error:", state.get("plan_error"))

    route = _route_after_validate(state)
    print("ROUTE:", route)

    if route == "stream_plan":
        state = await stream_plan_node(state)
        q = get_queue(session_id)
        event = await q.get()
        print("SSE EVENT:", event)


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Evaluate these suppliers quotes and generate a report"
    asyncio.run(main(message))
