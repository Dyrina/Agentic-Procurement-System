from __future__ import annotations

from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.mcp_server import mcp


@mcp.tool(description="Query purchase_history for avg unit_price_sen and avg delivery_days for the item")
async def query_history(item_id: str) -> dict:
    """MCP tool: compute historical price + delivery averages for risk flagging."""
    db = SupabaseRepository()
    history = db.get_purchase_history(item_id=item_id)

    if not history:
        return {"avg_unit_price_sen": 0.0, "avg_delivery_days": 0.0}

    avg_price = sum(row["unit_price_sen"] for row in history) / len(history)
    avg_delivery = sum(row["delivery_days"] for row in history) / len(history)

    return {
        "avg_unit_price_sen": round(avg_price, 2),
        "avg_delivery_days": round(avg_delivery, 2),
    }


async def query_history_handler(state: ProcurementState) -> ProcurementState:
    result = await query_history(state["item_id"])
    return {
        **state,
        "avg_unit_price_sen": result["avg_unit_price_sen"],
        "avg_delivery_days": result["avg_delivery_days"],
    }
