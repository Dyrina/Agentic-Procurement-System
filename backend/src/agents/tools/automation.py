"""
agents/tools/automation.py — PO write, PDF generation, and Storage upload.

Handles the post-approval automation:
  1. Write a purchase order row to Supabase.
  2. Generate a PDF document using ReportLab.
  3. Upload the PDF to Supabase Storage and return the public URL.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.core.config import get_settings
from src.database.client import SupabaseRepository


def _generate_po_pdf(
    po_number: str,
    supplier_name: str,
    item_name: str,
    quantity: int,
    unit_price_sen: int,
    total_amount_sen: int,
    date_str: str,
) -> bytes:
    """Generate a Purchase Order PDF and return as bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 60, "PURCHASE ORDER")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 100, f"PO Number: {po_number}")
    c.drawString(50, height - 118, f"Date: {date_str}")

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 160, "Supplier")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 178, supplier_name)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 220, "Order Details")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 238, f"Item: {item_name}")
    c.drawString(50, height - 256, f"Quantity: {quantity} units")
    c.drawString(50, height - 274, f"Unit Price: RM {unit_price_sen / 100:,.2f}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 310, f"TOTAL AMOUNT: RM {total_amount_sen / 100:,.2f}")

    c.setFont("Helvetica", 9)
    c.drawString(50, 50, "This is a system-generated purchase order.")

    c.save()
    return buffer.getvalue()


async def run_automation(state: dict) -> dict:
    """
    Post-approval automation: write PO to Supabase, generate PDF, upload to Storage.
    Returns updated state with po_pdf_url.
    Not registered as MCP tool — triggered only by approve endpoint.
    """
    settings = get_settings()
    db = SupabaseRepository()
    supabase_client = db._client

    suppliers = db.get_all_suppliers()
    recommended = next(
        (s for s in state.get("evaluated_suppliers", []) if s.get("is_recommended")),
        None,
    )
    if not recommended:
        raise ValueError("No recommended supplier found in state")

    supplier_info = next(
        (s for s in suppliers if s["supplier_id"] == recommended["supplier_id"]),
        None,
    )
    supplier_name = supplier_info["name"] if supplier_info else recommended.get("supplier_name", "Unknown")

    quantity = state.get("requested_qty", 1)
    unit_price_sen = recommended["unit_price_sen"]
    total_amount_sen = unit_price_sen * quantity
    item_id = state.get("item_id", "")
    item_name = state.get("item_name", "")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Write PO to Supabase
    po_record = db.create_purchase_order_full(
        supplier_id=recommended["supplier_id"],
        item_id=item_id,
        item_name=item_name,
        quantity=quantity,
        total_amount_sen=total_amount_sen,
        approved_by=state.get("user_id", "system"),
    )
    po_number = str(po_record.get("new_po_id", "PO-UNKNOWN"))

    # Generate PDF
    pdf_bytes = _generate_po_pdf(
        po_number=po_number,
        supplier_name=supplier_name,
        item_name=item_name,
        quantity=quantity,
        unit_price_sen=unit_price_sen,
        total_amount_sen=total_amount_sen,
        date_str=date_str,
    )

    # Upload to Supabase Storage
    filename = f"po_{po_number}.pdf"
    try:
        supabase_client.storage.from_("purchase-orders").upload(
            path=filename,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf"},
        )
        pdf_url = supabase_client.storage.from_("purchase-orders").get_public_url(filename)
    except Exception:
        pdf_url = ""

    # Update PO record with pdf_url
    db.update(
        "purchase_orders",
        filters={"new_po_id": po_number},
        data={"pdf_url": pdf_url},
    )

    db.write_audit_log(
        action_type="PURCHASE_ORDER_APPROVED",
        agent_name="automation_agent",
        decision_json={"po_number": po_number, "supplier_id": recommended["supplier_id"]},
    )

    return {**state, "status": "APPROVED", "po_pdf_url": pdf_url, "po_number": po_number}