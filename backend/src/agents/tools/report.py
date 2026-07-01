from __future__ import annotations

from src.core.state import ProcurementState
from src.mcp_server import mcp


@mcp.tool(description="Assemble executive summary, supplier comparison table, and decision explanation")
async def generate_report(
    evaluated_suppliers: list[dict],
    item_name: str,
    requested_qty: int,
    stock_sufficient: bool | None = None,
    current_stock: int | None = None,
) -> dict:
    """MCP tool: build markdown report from evaluated supplier data (no LLM call)."""
    recommended = next((s for s in evaluated_suppliers if s.get("is_recommended")), None)
    rec_name = recommended["supplier_name"] if recommended else "N/A"
    rec_price = recommended["unit_price_sen"] if recommended else 0
    rec_delivery = recommended["quoted_delivery_days"] if recommended else 0

    # Stock notice
    stock_notice = ""
    if stock_sufficient is not None:
        if stock_sufficient:
            stock_notice = f"\n> **Stock notice:** {current_stock} units in stock — sufficient for this order.\n"
        else:
            stock_notice = f"\n> ⚠️ **Stock notice:** Only {current_stock} units in stock — {requested_qty} requested. Procurement required.\n"

    # Comparison table
    table_rows = "\n".join(
        f"| {s['supplier_name']} | RM {s['unit_price_sen']/100:,.2f} | {s['quoted_delivery_days']} days | {s.get('payment_terms','N/A')} | {s['total_score']:.1f} | {'✅ Recommended' if s['is_recommended'] else ''} |"
        for s in sorted(evaluated_suppliers, key=lambda x: x["total_score"], reverse=True)
    )

    # Risk flags section
    risk_section = ""
    for s in evaluated_suppliers:
        if s.get("risk_flags"):
            risk_section += f"\n**{s['supplier_name']} risks:**\n"
            for flag in s["risk_flags"]:
                risk_section += f"- ⚠️ {flag}\n"

    # Score breakdown for recommended
    score_breakdown = ""
    if recommended:
        score_breakdown = (
            f"\n**Score breakdown for {rec_name}:**\n"
            f"- Price score: {recommended['price_score']:.1f} / 100 (weight 55%)\n"
            f"- Delivery score: {recommended['delivery_score']:.1f} / 100 (weight 30%)\n"
            f"- Payment terms score: {recommended['payment_terms_score']:.1f} / 100 (weight 15%)\n"
            f"- **Total: {recommended['total_score']:.1f} / 100**\n"
        )

    markdown = f"""## Procurement Evaluation Report

**Item:** {item_name}
**Requested Quantity:** {requested_qty} units
{stock_notice}
---

### Supplier Comparison

| Supplier | Unit Price | Delivery | Payment Terms | Score | Recommendation |
|----------|-----------|----------|---------------|-------|----------------|
{table_rows}

---

### Recommendation

**{rec_name}** is recommended with a total score of **{recommended['total_score']:.1f}/100** (if available).

Estimated unit price: RM {rec_price/100:,.2f} | Delivery: {rec_delivery} days
{score_breakdown}
{risk_section}
---

*Evaluation performed by AI Procurement Operations System. Approve below to generate a Purchase Order.*
"""
    return {"report_markdown": markdown.strip()}


async def generate_report_handler(state: ProcurementState) -> ProcurementState:
    result = await generate_report.fn(
        evaluated_suppliers=state.get("evaluated_suppliers", []),
        item_name=state.get("item_name", ""),
        requested_qty=state.get("requested_qty", 0),
        stock_sufficient=state.get("stock_sufficient"),
        current_stock=state.get("current_stock"),
    )
    return {**state, "report_markdown": result["report_markdown"]}
