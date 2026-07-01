from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as chat_router

# Import mcp_server to trigger @mcp.tool() registrations in all tool modules
import src.mcp_server  # noqa: F401
import src.agents.tools.stock  # noqa: F401
import src.agents.tools.rfq  # noqa: F401
import src.agents.tools.quotes  # noqa: F401
import src.agents.tools.history  # noqa: F401
import src.agents.tools.evaluation  # noqa: F401
import src.agents.tools.report  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from src.core.config import get_settings
    get_settings()  # raises early if SUPABASE_URL / KEY missing
    yield


app = FastAPI(
    title="AI Procurement Operations API",
    description="Multi-Agent System backend. Powered by LangGraph + Gemini + FastMCP.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": "ai-procurement-ops", "status": "healthy", "version": "0.2.0"}


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}