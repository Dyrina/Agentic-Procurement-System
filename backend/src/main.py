from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as chat_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from src.core.config import get_checkpoint_pool, get_checkpointer, get_settings

    get_settings()  # raises early if required env vars are missing

    pool = get_checkpoint_pool()
    await pool.open()
    await get_checkpointer().setup()  # idempotent — creates the checkpoint tables if needed
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(
    title="AI Procurement Operations API",
    description="Multi-Agent System backend. Powered by LangGraph + Gemini.",
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
