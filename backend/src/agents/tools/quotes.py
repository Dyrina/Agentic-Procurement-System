from __future__ import annotations

import asyncio
import io
from typing import Any

import pypdf
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.state import ProcurementState
from src.mcp_server import mcp
from src.services.gmail import fetch_replies, get_gmail_service

_POLL_INTERVAL = 15  # seconds
_DEFAULT_TIMEOUT = 300  # 5 minutes


@mcp.tool(description="Poll Gmail every 15s for supplier replies; proceed when all replied or 5-min timeout")
async def wait_for_quotes(
    supplier_emails: list[str],
    rfq_sent_at: str,
    timeout_seconds: int = _DEFAULT_TIMEOUT,
) -> dict:
    """MCP tool: block until all suppliers reply or timeout."""
    settings = get_settings()
    service = get_gmail_service(settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH)

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        replies = fetch_replies(service, from_emails=supplier_emails, since_timestamp=rfq_sent_at)
        replied_from = {r["from"].split("<")[-1].strip(">").strip() for r in replies}
        if all(email in replied_from for email in supplier_emails):
            return {"all_replied": True}
        await asyncio.sleep(_POLL_INTERVAL)

    return {"all_replied": False}


async def wait_for_quotes_handler(state: ProcurementState) -> ProcurementState:
    result = await wait_for_quotes.fn(
        supplier_emails=state["supplier_emails"],
        rfq_sent_at=state["rfq_sent_at"],
    )
    return {**state, "all_replied": result["all_replied"]}


class _QuoteExtracted(BaseModel):
    supplier_name: str
    unit_price_sen: int
    quoted_delivery_days: int
    payment_terms: str


class _QuotesOutput(BaseModel):
    quotes: list[_QuoteExtracted]


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_quotes_with_gemini(text: str, api_key: str) -> list[dict[str, Any]]:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)
    structured = llm.with_structured_output(_QuotesOutput)
    prompt = (
        "Extract procurement quote data from this supplier communication.\n\n"
        f"Text:\n{text}\n\n"
        "Extract all quotes found. For each quote:\n"
        "- supplier_name: company name\n"
        "- unit_price_sen: unit price in sen (Malaysian Ringgit cents; RM 1 = 100 sen; e.g. RM 3,950.00 = 395000)\n"
        "- quoted_delivery_days: delivery time as integer number of days\n"
        "- payment_terms: e.g. 'Net-30', 'Net-60', 'Net-45', 'Net-15', 'Immediate'"
    )
    result = structured.invoke(prompt)
    return [q.model_dump() for q in result.quotes]


@mcp.tool(description="Fetch reply emails + PDF attachments via Gmail; extract structured quote data with Gemini")
async def extract_quotes(supplier_emails: list[str], rfq_sent_at: str) -> dict:
    """MCP tool: read Gmail replies, parse PDFs/bodies, extract structured quotes."""
    settings = get_settings()
    service = get_gmail_service(settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH)

    replies = fetch_replies(service, from_emails=supplier_emails, since_timestamp=rfq_sent_at)
    if not replies:
        raise ValueError("No supplier replies found in Gmail")

    db_suppliers = None  # lazy-load only if needed for supplier_id lookup

    all_quotes: list[dict[str, Any]] = []
    for reply in replies:
        # Prefer PDF attachment text, fall back to email body
        if reply["attachments"]:
            text = _extract_text_from_pdf(reply["attachments"][0]["data"])
        else:
            text = reply["body_text"]

        if not text.strip():
            continue

        quotes = _parse_quotes_with_gemini(text, settings.GOOGLE_API_KEY)
        # Enrich each quote with supplier_id by matching supplier_name to DB
        if db_suppliers is None:
            from src.database.client import SupabaseRepository
            db = SupabaseRepository()
            db_suppliers = db.get_all_suppliers()

        for q in quotes:
            matched = next(
                (s for s in db_suppliers if s["name"].lower() in q["supplier_name"].lower()),
                None,
            )
            q["supplier_id"] = matched["supplier_id"] if matched else "UNKNOWN"
            all_quotes.append(q)

    return {"extracted_quotes": all_quotes}


async def extract_quotes_handler(state: ProcurementState) -> ProcurementState:
    result = await extract_quotes.fn(
        supplier_emails=state["supplier_emails"],
        rfq_sent_at=state["rfq_sent_at"],
    )
    return {**state, "extracted_quotes": result["extracted_quotes"]}
