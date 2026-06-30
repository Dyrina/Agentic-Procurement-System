"""
database/client.py — Supabase CRUD wrapper.

Provides a thin abstraction layer over the Supabase Python client so that
the rest of the application doesn't couple directly to the raw SDK calls.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from supabase import Client

from src.core.config import get_supabase_client


class SupabaseRepository:
    """Generic CRUD helper for any Supabase table."""

    def __init__(self, client: Client | None = None) -> None:
        self._client: Client = client or get_supabase_client()

    # ── Generic helpers ─────────────────────────────────────────────────

    def select(
        self,
        table: str,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """SELECT rows from *table*, optionally filtered by equality conditions."""
        query = self._client.table(table).select(columns)
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        response = query.execute()
        return response.data

    def insert(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        """INSERT a single row and return the created record."""
        response = self._client.table(table).insert(data).execute()
        return response.data[0] if response.data else {}

    def update(
        self,
        table: str,
        filters: dict[str, Any],
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """UPDATE rows matching *filters* with *data*."""
        query = self._client.table(table).update(data)
        for col, val in filters.items():
            query = query.eq(col, val)
        response = query.execute()
        return response.data

    def delete(self, table: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """DELETE rows matching *filters*."""
        query = self._client.table(table).delete()
        for col, val in filters.items():
            query = query.eq(col, val)
        response = query.execute()
        return response.data

    # ── Domain-specific convenience methods ─────────────────────────────

    def get_supplier(self, supplier_id: str) -> dict[str, Any] | None:
        """Fetch a single supplier by ID."""
        rows = self.select("suppliers", filters={"supplier_id": supplier_id})
        return rows[0] if rows else None

    def get_all_suppliers(self) -> list[dict[str, Any]]:
        """Return every supplier row."""
        return self.select("suppliers")

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        """Fetch a single inventory item by ID."""
        rows = self.select("items", filters={"item_id": item_id})
        return rows[0] if rows else None

    def get_purchase_history(
        self,
        item_id: str | None = None,
        supplier_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return purchase-history rows, optionally filtered."""
        filters: dict[str, Any] = {}
        if item_id:
            filters["item_id"] = item_id
        if supplier_id:
            filters["supplier_id"] = supplier_id
        return self.select("purchase_history", filters=filters or None)

    def create_purchase_order(
        self,
        supplier_id: str,
        total_amount_sen: int,
        approved_by: str,
        status: str = "APPROVED",
    ) -> dict[str, Any]:
        """Insert a new purchase order and return the created record."""
        return self.insert(
            "purchase_orders",
            {
                "new_po_id": str(uuid4()),
                "supplier_id": supplier_id,
                "total_amount_sen": total_amount_sen,
                "status": status,
                "approved_by": approved_by,
            },
        )

    def write_audit_log(
        self,
        action_type: str,
        agent_name: str,
        decision_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an entry to the audit_logs table."""
        return self.insert(
            "audit_logs",
            {
                "log_id": str(uuid4()),
                "action_type": action_type,
                "agent_name": agent_name,
                "decision_json": decision_json or {},
            },
        )


# Module-level convenience instance
db = SupabaseRepository()
