"""
core/config.py — Application configuration & Supabase client factory.

Reads SUPABASE_URL and SUPABASE_KEY from environment variables and
exposes a singleton Supabase client for the rest of the application.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

# Load .env when running outside Docker (local dev)
load_dotenv()


class Settings:
    """Centralised application settings pulled from environment variables."""

    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GMAIL_CREDENTIALS_PATH: str = os.environ.get("GMAIL_CREDENTIALS_PATH", "credentials.json")
    GMAIL_TOKEN_PATH: str = os.environ.get("GMAIL_TOKEN_PATH", "token.json")
    GMAIL_SENDER_EMAIL: str = os.environ.get("GMAIL_SENDER_EMAIL", "")

    def validate(self) -> None:
        """Raise early if critical env vars are missing."""
        missing: list[str] = []
        if not self.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not self.SUPABASE_KEY:
            missing.append("SUPABASE_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return (and cache) the application settings singleton."""
    settings = Settings()
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Return (and cache) an authenticated Supabase client."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
