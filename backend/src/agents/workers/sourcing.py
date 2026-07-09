"""agents/workers/sourcing.py — Sourcing specialist: RFQ send/wait/extract + supplier resolution.

Replaces agents/tools/rfq.py + agents/tools/quotes.py. The step ordering here (send -> wait ->
extract) is mechanical, not a judgment call, so this worker is a deterministic state-machine
node rather than an LLM tool-calling loop — the one real decision point (what to do about a
partial-reply timeout) is delegated straight to the human via interrupt(), not to an LLM.

Supplier identity is resolved by the reply's sender email against suppliers.contact_email —
fetch_replies already filtered replies to known from_emails, so that address is a trustworthy
key. This replaces the old fuzzy substring match of a PDF-parsed company name against
suppliers.name (which silently tagged mismatches "UNKNOWN").
"""

from __future__ import annotations

import asyncio
import logging
import io
from datetime import datetime, timezone
from typing import Any

import pypdf
from langgraph.types import interrupt
from pydantic import BaseModel

from src.agents.workers import _build_llm, _cancel_requested, _extract_text, _format_error
from src.core.config import get_settings
from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.services.gmail import fetch_replies, get_gmail_service, send_email

_POLL_INTERVAL = 15  # seconds
_DEFAULT_TIMEOUT = 300  # 5 minutes


logger = logging.getLogger(__name__)


def _history_entry(summary: str) -> dict[str, str]:
    return {"worker": "sourcing", "summary": summary}


async def _draft_rfq_email(item_name: str, requested_qty: int) -> str:
    """Use Gemini to draft an RFQ email body in HTML."""
    prompt = (
        f"Draft a professional Request for Quotation (RFQ) email body in HTML.\n"
        f"Item: {item_name}\nQuantity: {requested_qty} units\n\n"
        f"The email must ask for:\n- Unit price (please specify in Malaysian Ringgit)\n"
        f"- Estimated delivery timeline (days)\n- Payment terms (e.g. Net-30, Net-60)\n"
        f"- PDF quotation attachment preferred\n\n"
        f"Reply deadline: within 24 hours. Be concise and professional.\n"
        f"Return only the HTML email body, no subject line, no preamble."
    )
    response = await _build_llm().ainvoke(prompt)
    return _extract_text(response.content)


async def send_rfqs(item_name: str, item_category: str, requested_qty: int) -> dict:
    """Draft an RFQ with Gemini and send it to relevant suppliers based on category."""
    settings = get_settings()
    db = SupabaseRepository()

    suppliers = await asyncio.to_thread(db.get_suppliers_by_category, item_category)
    if not suppliers:
        raise ValueError(f"No suppliers found for category: {item_category}")

    email_body = await _draft_rfq_email(item_name, requested_qty)
    subject = f"Request for Quotation — {item_name} (Qty: {requested_qty})"

    service = await asyncio.to_thread(
        get_gmail_service, settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH
    )
    sent_to: list[str] = []
    for supplier in suppliers:
        await asyncio.to_thread(
            send_email,
            service,
            to=supplier["contact_email"],
            subject=subject,
            body_html=email_body,
        )
        sent_to.append(supplier["contact_email"])

    return {"rfq_sent_at": datetime.now(timezone.utc).isoformat(), "supplier_emails": sent_to}


async def wait_for_quotes(
    supplier_emails: list[str], rfq_sent_at: str, timeout_seconds: int = _DEFAULT_TIMEOUT
) -> dict:
    """Poll Gmail until every supplier has replied or timeout_seconds elapses."""
    settings = get_settings()
    service = await asyncio.to_thread(
        get_gmail_service, settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH
    )

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while True:
        replies = await asyncio.to_thread(
            fetch_replies, service, from_emails=supplier_emails, since_timestamp=rfq_sent_at
        )
        replied_from = {r["from"].split("<")[-1].strip(">").strip().lower() for r in replies}
        pending = [e for e in supplier_emails if e.lower() not in replied_from]
        if not pending:
            return {"all_replied": True, "pending_emails": []}
        if asyncio.get_event_loop().time() >= deadline:
            return {"all_replied": False, "pending_emails": pending}
        await asyncio.sleep(_POLL_INTERVAL)


async def send_reminder_email(pending_emails: list[str]) -> dict:
    """Send a short follow-up to suppliers who haven't replied yet."""
    settings = get_settings()
    service = await asyncio.to_thread(
        get_gmail_service, settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH
    )
    body_html = "<p>Following up on our RFQ — could you send your quote when you have a chance?</p>"
    for email in pending_emails:
        await asyncio.to_thread(
            send_email, service, to=email, subject="Reminder: Request for Quotation", body_html=body_html
        )
    return {"reminded_emails": list(pending_emails)}


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


async def _parse_quotes_with_gemini(text: str) -> list[dict[str, Any]]:
    # Money path — a misread sen amount here becomes a wrong PO, so this gets the smart tier.
    structured = _build_llm("smart").with_structured_output(_QuotesOutput)
    prompt = (
        "Extract procurement quote data from this supplier communication.\n\n"
        f"Text:\n{text}\n\n"
        "Extract all quotes found. For each quote:\n"
        "- supplier_name: company name\n"
        "- unit_price_sen: unit price in sen (Malaysian Ringgit cents; RM 1 = 100 sen)\n"
        "- quoted_delivery_days: delivery time as integer number of days\n"
        "- payment_terms: e.g. 'Net-30', 'Net-60', 'Net-45', 'Net-15', 'Immediate'"
    )
    result = await structured.ainvoke(prompt)
    return [q.model_dump() for q in result.quotes]


async def extract_quotes(supplier_emails: list[str], rfq_sent_at: str) -> dict:
    """Parse supplier replies into structured quotes, resolving supplier_id by sender email."""
    settings = get_settings()
    service = await asyncio.to_thread(
        get_gmail_service, settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH
    )
    replies = await asyncio.to_thread(
        fetch_replies, service, from_emails=supplier_emails, since_timestamp=rfq_sent_at
    )
    if not replies:
        raise ValueError("No supplier replies found in Gmail")

    db = SupabaseRepository()
    suppliers = await asyncio.to_thread(db.get_all_suppliers)
    suppliers_by_email = {s["contact_email"].lower(): s for s in suppliers}

    all_quotes: list[dict[str, Any]] = []
    for reply in replies:
        # fetch_replies already filtered replies to from_emails=supplier_emails, so the sender
        # address is a trustworthy key — no need to fuzzy-match a PDF-parsed company name.
        sender_email = reply["from"].split("<")[-1].strip(">").strip().lower()
        supplier = suppliers_by_email.get(sender_email)
        supplier_id = supplier["supplier_id"] if supplier else "UNKNOWN"
        supplier_name = supplier["name"] if supplier else "Unknown"

        text = (
            await asyncio.to_thread(_extract_text_from_pdf, reply["attachments"][0]["data"])
            if reply["attachments"]
            else reply["body_text"]
        )
        if not text.strip():
            continue

        for q in await _parse_quotes_with_gemini(text):
            q["supplier_id"] = supplier_id
            q["supplier_name"] = supplier_name
            all_quotes.append(q)

    return {"extracted_quotes": all_quotes}


async def sourcing_node(state: ProcurementState) -> ProcurementState:
    """Deterministically advance the RFQ -> wait -> extract pipeline one stage per call."""
    history = state.get("supervisor_history", [])
    try:
        if not state.get("rfq_sent_at"):
            item_category = state.get("item_category", "General")
            result = await send_rfqs(state["item_name"], item_category, state["requested_qty"])
            return {
                **state,
                **result,
                "supervisor_history": [
                    *history,
                    _history_entry(f"RFQ sent to {len(result['supplier_emails'])} suppliers"),
                ],
            }

        if not state.get("extracted_quotes"):
            wait_result = await wait_for_quotes(state["supplier_emails"], state["rfq_sent_at"])
            if wait_result["all_replied"]:
                quotes_result = await extract_quotes(state["supplier_emails"], state["rfq_sent_at"])
                return {
                    **state,
                    "all_replied": True,
                    "pending_emails": [],
                    **quotes_result,
                    "supervisor_history": [
                        *history,
                        _history_entry(
                            f"{len(quotes_result['extracted_quotes'])} quotes extracted"
                        ),
                    ],
                }
            return {
                **state,
                "all_replied": False,
                "pending_emails": wait_result["pending_emails"],
                "needs_clarification": True,
                "clarification_payload": {
                    "type": "sourcing_timeout",
                    "message": (
                        f"{len(wait_result['pending_emails'])} of {len(state['supplier_emails'])} "
                        "suppliers have not replied yet."
                    ),
                    "pending_emails": wait_result["pending_emails"],
                    "options": ["proceed_partial", "extend_wait", "send_reminder"],
                },
            }

        # extracted_quotes already present — nothing left for this worker to do.
        return {
            **state,
            "supervisor_history": [*history, _history_entry("quotes already extracted")],
        }
    except Exception as exc:
        logger.exception("sourcing worker failed")
        return {
            **state,
            "error": _format_error(exc),
            "supervisor_history": [*history, _history_entry(f"FAILED: {exc}")],
        }


async def sourcing_await_node(state: ProcurementState) -> ProcurementState:
    """Pause node: interrupt()s on timeout, resumes with the user's escalation choice."""
    if not state.get("needs_clarification"):
        return state
    answer = interrupt(state["clarification_payload"])
    base = {**state, "needs_clarification": False, "clarification_payload": None}
    if _cancel_requested(answer):
        return {**base, "cancelled": True}

    action = answer.get("action")
    if action == "proceed_partial":
        quotes_result = await extract_quotes(state["supplier_emails"], state["rfq_sent_at"])
        return {**base, "all_replied": True, "pending_emails": [], **quotes_result}
    if action == "send_reminder":
        await send_reminder_email(state["pending_emails"])
    # "extend_wait" and "send_reminder" both just loop back to wait_for_quotes again.
    return base
