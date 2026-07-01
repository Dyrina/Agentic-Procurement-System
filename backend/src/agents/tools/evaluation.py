from __future__ import annotations

from src.core.state import ProcurementState
from src.mcp_server import mcp
from src.services.scoring import score_suppliers


@mcp.tool(description="Deterministic weighted scoring: Price 55%, Delivery 30%, Payment Terms 15%, plus risk flags vs historical averages")
async def evaluate_suppliers(
    extracted_quotes: list[dict],
    avg_unit_price_sen: float,
    avg_delivery_days: float,
) -> dict:
    """MCP tool: run scoring engine over extracted quotes."""
    if not extracted_quotes:
        raise ValueError("No quotes to evaluate")
    scored = score_suppliers(
        quotes=extracted_quotes,
        avg_historical_price_sen=avg_unit_price_sen,
        avg_historical_delivery_days=avg_delivery_days,
    )
    return {"evaluated_suppliers": scored}


async def evaluate_suppliers_handler(state: ProcurementState) -> ProcurementState:
    result = await evaluate_suppliers.fn(
        extracted_quotes=state.get("extracted_quotes", []),
        avg_unit_price_sen=state.get("avg_unit_price_sen", 0.0),
        avg_delivery_days=state.get("avg_delivery_days", 0.0),
    )
    return {**state, "evaluated_suppliers": result["evaluated_suppliers"]}
