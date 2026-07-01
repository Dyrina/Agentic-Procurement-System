from __future__ import annotations

from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.mcp_server import mcp


@mcp.tool(description="Query items.current_stock vs requested quantity")
async def check_stock(item_name: str, requested_qty: int) -> dict:
    """MCP tool: look up item by name and compare stock to requested quantity."""
    db = SupabaseRepository()
    rows = db.select("items", filters={"name": item_name})
    if not rows:
        # Try case-insensitive partial match
        all_items = db.select("items")
        rows = [r for r in all_items if item_name.lower() in r["name"].lower()]
    if not rows:
        raise ValueError(f"Item '{item_name}' not found in inventory")
    item = rows[0]
    return {
        "item_id": item["item_id"],
        "current_stock": item["current_stock"],
        "stock_sufficient": item["current_stock"] >= requested_qty,
    }


async def check_stock_handler(state: ProcurementState) -> ProcurementState:
    """Execute node handler: reads item_name + requested_qty from state, writes result back."""
    result = await check_stock.fn(state["item_name"], state["requested_qty"])
    return {
        **state,
        "item_id": result["item_id"],
        "current_stock": result["current_stock"],
        "stock_sufficient": result["stock_sufficient"],
    }
