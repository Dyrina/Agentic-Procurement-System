"""
mcp_server.py — FastMCP server instance.

Import mcp here and decorate tools with @mcp.tool() in each agents/tools/*.py.
Import mcp_server in main.py to trigger tool registration at startup.
"""

from fastmcp import FastMCP

mcp = FastMCP("procurement")

TOOL_DESCRIPTIONS: dict[str, str] = {
    "check_stock": "Query items.current_stock vs requested quantity to determine if stock is sufficient",
    "send_rfqs": "Draft RFQ email via Gemini and send to all registered supplier contact_email addresses via Gmail API",
    "wait_for_quotes": "Poll Gmail every 15s for replies from supplier emails since RFQ was sent; proceed when all replied or 5-minute timeout",
    "extract_quotes": "Fetch reply emails + PDF attachments via Gmail; use pypdf and Gemini to extract structured quote data (unit_price_sen, delivery_days, payment_terms)",
    "query_history": "Query purchase_history for avg unit_price_sen and avg delivery_days for the requested item",
    "evaluate_suppliers": "Run deterministic weighted scoring (Price 55%, Delivery 30%, Payment Terms 15%) and flag risks vs historical averages",
    "generate_report": "Assemble executive summary, supplier comparison table, and decision explanation from evaluated supplier data",
}

TOOL_REGISTRY_KEYS = set(TOOL_DESCRIPTIONS.keys())

ORDERING_CONSTRAINTS: list[tuple[str, str]] = [
    ("send_rfqs", "wait_for_quotes"),
    ("wait_for_quotes", "extract_quotes"),
    ("extract_quotes", "evaluate_suppliers"),
    ("query_history", "evaluate_suppliers"),
    ("evaluate_suppliers", "generate_report"),
]


def validate_plan(plan: list[str]) -> str | None:
    """Return error string if plan is invalid, None if valid."""
    for step in plan:
        if step not in TOOL_REGISTRY_KEYS:
            valid = sorted(TOOL_REGISTRY_KEYS)
            return f"Unknown step '{step}'. Valid steps: {valid}"

    for before, after in ORDERING_CONSTRAINTS:
        if before in plan and after in plan:
            if plan.index(before) >= plan.index(after):
                return f"'{before}' must appear before '{after}' in the plan"

    return None
