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
        self._explicit_client = client

    @property
    def _client(self) -> Client:
        """Lazy-load the Supabase client on first use, not at import time."""
        if self._explicit_client is not None:
            return self._explicit_client
        return get_supabase_client()

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

    def rpc(self, fn_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Call a Postgres function (e.g. search_items_by_name) and return its rows."""
        return self._client.rpc(fn_name, params).execute().data

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

    def create_evaluation(self, session_id: str, user_id: str) -> dict[str, Any]:
        """Insert a new evaluation session row."""
        return self.insert(
            "evaluations",
            {"session_id": session_id, "user_id": user_id, "status": "PLANNING"},
        )

    def get_evaluation(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a single evaluation by session ID."""
        rows = self.select("evaluations", filters={"session_id": session_id})
        return rows[0] if rows else None

    def update_evaluation(self, session_id: str, **fields: Any) -> list[dict[str, Any]]:
        """Update evaluation fields. Pass keyword args for each column to update."""
        from datetime import datetime, timezone
        return self.update(
            "evaluations",
            filters={"session_id": session_id},
            data={**fields, "updated_at": datetime.now(timezone.utc).isoformat()},
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

    def create_purchase_order_full(
        self,
        supplier_id: str,
        item_id: str,
        quantity: int,
        total_amount_sen: int,
        approved_by: str,
        pdf_url: str = "",
    ) -> dict[str, Any]:
        """Insert a purchase order with item, quantity, and PDF URL."""
        return self.insert(
            "purchase_orders",
            {
                "supplier_id": supplier_id,
                "item_id": item_id,
                "quantity": quantity,
                "total_amount_sen": total_amount_sen,
                "status": "APPROVED",
                "approved_by": approved_by,
                "pdf_url": pdf_url,
            },
        )

# Module-level convenience instance
db = SupabaseRepository()
