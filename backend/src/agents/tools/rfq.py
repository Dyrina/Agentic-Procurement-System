from __future__ import annotations

from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import get_settings
from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.mcp_server import mcp
from src.services.gmail import get_gmail_service, send_email


def _draft_rfq_email(item_name: str, requested_qty: int, api_key: str) -> str:
    """Use Gemini to draft an RFQ email body in HTML."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)
    prompt = (
        f"Draft a professional Request for Quotation (RFQ) email body in HTML.\n"
        f"Item: {item_name}\n"
        f"Quantity: {requested_qty} units\n\n"
        f"The email must ask for:\n"
        f"- Unit price (please specify in Malaysian Ringgit)\n"
        f"- Estimated delivery timeline (days)\n"
        f"- Payment terms (e.g. Net-30, Net-60)\n"
        f"- PDF quotation attachment preferred\n\n"
        f"Reply deadline: within 24 hours. Be concise and professional.\n"
        f"Return only the HTML email body, no subject line, no preamble."
    )
    response = llm.invoke(prompt)
    return response.content


@mcp.tool(description="Draft RFQ email via Gemini and send to all registered supplier emails")
async def send_rfqs(item_name: str, requested_qty: int) -> dict:
    """MCP tool: draft RFQ with Gemini, send to all suppliers in DB via Gmail."""
    settings = get_settings()
    db = SupabaseRepository()

    suppliers = db.get_all_suppliers()
    if not suppliers:
        raise ValueError("No suppliers registered in database")

    email_body = _draft_rfq_email(item_name, requested_qty, settings.GOOGLE_API_KEY)
    subject = f"Request for Quotation — {item_name} (Qty: {requested_qty})"

    service = get_gmail_service(settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH)
    sent_to: list[str] = []
    for supplier in suppliers:
        send_email(service, to=supplier["contact_email"], subject=subject, body_html=email_body)
        sent_to.append(supplier["contact_email"])

    rfq_sent_at = datetime.now(timezone.utc).isoformat()
    return {"rfq_sent_at": rfq_sent_at, "supplier_emails": sent_to}


async def send_rfqs_handler(state: ProcurementState) -> ProcurementState:
    result = await send_rfqs(state["item_name"], state["requested_qty"])
    return {**state, "rfq_sent_at": result["rfq_sent_at"], "supplier_emails": result["supplier_emails"]}
