from __future__ import annotations

import asyncio

from src.database.client import SupabaseRepository


async def query_history(item_id: str) -> dict:
    """Compute historical price + delivery averages for risk flagging."""
    db = SupabaseRepository()
    history = await asyncio.to_thread(db.get_purchase_history, item_id=item_id)

    if not history:
        return {"avg_unit_price_sen": 0.0, "avg_delivery_days": 0.0}

    avg_price = sum(row["unit_price_sen"] for row in history) / len(history)
    avg_delivery = sum(row["delivery_days"] for row in history) / len(history)

    return {
        "avg_unit_price_sen": round(avg_price, 2),
        "avg_delivery_days": round(avg_delivery, 2),
    }
