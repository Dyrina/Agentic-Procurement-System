"""
main.py — FastAPI application entrypoint.

Registers API routers and provides health-check / metadata endpoints.
Run with:  uvicorn src.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as evaluations_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan events (startup / shutdown)."""
    # Startup — validate that critical env vars are present early.
    from src.core.config import get_settings

    get_settings()  # will raise if SUPABASE_URL / KEY are missing
    yield
    # Shutdown — nothing to clean up for now.


app = FastAPI(
    title="AI Procurement Operations API",
    description=(
        "Multi-Agent System backend for automating the Source-to-Order "
        "procurement lifecycle. Powered by LangGraph + Gemini."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(evaluations_router)


# ── Root health-check ──────────────────────────────────────────────────────


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Root health-check endpoint."""
    return {
        "service": "ai-procurement-ops",
        "status": "healthy",
        "version": "0.1.0",
    }


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Explicit health endpoint for container orchestrators."""
    return {"status": "ok"}
