# AI Procurement Operations Multi-Agent System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent agentic procurement system: LangGraph plan-then-execute Manager Agent, FastMCP worker tools, SSE chat interface, Gmail integration, deterministic scoring, Supabase persistence, and post-approval PDF purchase order generation.

**Architecture:** Procurement Manager Agent (Gemini via LangGraph) generates a JSON plan from a 7-tool registry, validates ordering rules, streams the plan to the user via SSE, then executes each tool in sequence. Worker tools are Python async functions registered with an in-process FastMCP server. Automation Agent (PO + PDF) fires post-approval outside the LLM plan.

**Tech Stack:** FastAPI, LangGraph, FastMCP, Gemini (`langchain-google-genai`, model `gemini-2.0-flash`), Gmail API (`google-api-python-client`), `pypdf`, Supabase, `sse-starlette`, `reportlab`, `pytest`, `pytest-asyncio`

---

## File Map

**Create:**
- `backend/src/core/state.py` — `ProcurementState` TypedDict (shared state flowing through all nodes/tools)
- `backend/src/api/sse.py` — per-session `asyncio.Queue`, SSE event helpers
- `backend/src/services/scoring.py` — deterministic weighted scoring engine
- `backend/src/services/gmail.py` — Gmail API wrapper (send + fetch)
- `backend/src/mcp_server.py` — FastMCP instance; all tools imported here for registration
- `backend/src/agents/tools/__init__.py` — empty
- `backend/src/agents/tools/stock.py` — `check_stock` tool + handler
- `backend/src/agents/tools/rfq.py` — `send_rfqs` tool + handler
- `backend/src/agents/tools/quotes.py` — `wait_for_quotes` + `extract_quotes` tools + handlers
- `backend/src/agents/tools/history.py` — `query_history` tool + handler
- `backend/src/agents/tools/evaluation.py` — `evaluate_suppliers` tool + handler
- `backend/src/agents/tools/report.py` — `generate_report` tool + handler
- `backend/src/agents/tools/automation.py` — PO write + PDF generation + Storage upload
- `backend/src/agents/manager.py` — LangGraph graph (plan → validate → stream_plan → execute → END)
- `backend/tests/__init__.py`
- `backend/tests/test_scoring.py`
- `backend/tests/test_tools.py`
- `backend/tests/test_manager.py`
- `backend/tests/test_sse.py`

**Modify:**
- `backend/pyproject.toml` — add `fastmcp`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `sse-starlette`, `reportlab`, `pytest`, `pytest-asyncio`
- `backend/.env.example` — add Gmail OAuth env vars
- `backend/src/core/config.py` — add `GMAIL_CREDENTIALS_PATH`, `GMAIL_TOKEN_PATH`, `GMAIL_SENDER_EMAIL`
- `backend/schema.sql` — add `delivery_days` to `purchase_history`; add `evaluations` table; alter `purchase_orders`
- `backend/src/database/client.py` — add `create_evaluation`, `update_evaluation`, `get_evaluation`
- `backend/src/api/routes.py` — replace all old endpoints with `/chat` SSE endpoints
- `backend/src/main.py` — swap router import; import `mcp_server` to trigger tool registration

**Delete logic (not files):**
- `backend/src/agents/graph.py` — kept as-is but unused (old sequential graph); `manager.py` is the new graph

---

## Task 1: Update Dependencies and Config

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/.env.example`
- Modify: `backend/src/core/config.py`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

Replace the `dependencies` block in `backend/pyproject.toml`:

```toml
[project]
name = "agentic-procurement-system"
version = "0.1.0"
description = "AI Procurement Operations Multi-Agent System Backend"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "supabase>=2.15",
    "pydantic>=2.11",
    "langgraph>=0.5",
    "langchain-google-genai>=2.1",
    "langchain-core>=0.3",
    "python-multipart>=0.0.20",
    "pypdf>=5.5",
    "python-dotenv>=1.1",
    "fastmcp>=2.0",
    "google-api-python-client>=2.100",
    "google-auth-httplib2>=0.2",
    "google-auth-oauthlib>=1.2",
    "sse-starlette>=2.1",
    "reportlab>=4.2",
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

- [ ] **Step 2: Update .env.example with Gmail vars**

Append to `backend/.env.example`:

```
# Gmail OAuth (download credentials.json from Google Cloud Console)
GMAIL_CREDENTIALS_PATH=credentials.json
GMAIL_TOKEN_PATH=token.json
GMAIL_SENDER_EMAIL=your-gmail@gmail.com
```

- [ ] **Step 3: Add Gmail settings to config.py**

Add three new fields to the `Settings` class in `backend/src/core/config.py`:

```python
class Settings:
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GMAIL_CREDENTIALS_PATH: str = os.environ.get("GMAIL_CREDENTIALS_PATH", "credentials.json")
    GMAIL_TOKEN_PATH: str = os.environ.get("GMAIL_TOKEN_PATH", "token.json")
    GMAIL_SENDER_EMAIL: str = os.environ.get("GMAIL_SENDER_EMAIL", "")
```

- [ ] **Step 4: Install dependencies**

```bash
cd backend && uv sync
```

Expected: packages install without error.

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/.env.example backend/src/core/config.py
git commit -m "feat: add fastmcp, gmail, sse-starlette, reportlab dependencies"
```

---

## Task 2: Update Database Schema

**Files:**
- Modify: `backend/schema.sql`

- [ ] **Step 1: Add schema changes**

Append to the end of `backend/schema.sql` (after the seed data block):

```sql
-- ============================================================
-- Schema v2 migrations
-- ============================================================

-- Add delivery_days to purchase_history (missing from v1)
ALTER TABLE purchase_history ADD COLUMN IF NOT EXISTS delivery_days INT NOT NULL DEFAULT 0;

-- Update seed data with delivery_days
UPDATE purchase_history SET delivery_days = 7  WHERE po_id = 'PO-2025-001';
UPDATE purchase_history SET delivery_days = 12 WHERE po_id = 'PO-2025-002';
UPDATE purchase_history SET delivery_days = 21 WHERE po_id = 'PO-2025-003';

-- Add item_id, quantity, pdf_url to purchase_orders
ALTER TABLE purchase_orders ADD COLUMN IF NOT EXISTS item_id  TEXT REFERENCES items(item_id);
ALTER TABLE purchase_orders ADD COLUMN IF NOT EXISTS quantity  INT;
ALTER TABLE purchase_orders ADD COLUMN IF NOT EXISTS pdf_url  TEXT;

-- Evaluations: persists full pipeline state per chat session
CREATE TABLE IF NOT EXISTS evaluations (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PLANNING',
    plan_json       JSONB,
    current_step    TEXT,
    state_json      JSONB,
    report_markdown TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
```

- [ ] **Step 2: Apply to Supabase**

Run the new SQL block in the Supabase SQL editor (Project → SQL Editor → paste the migration block above → Run).

Verify: `evaluations` table appears in the Table Editor; `purchase_history` shows a `delivery_days` column.

- [ ] **Step 3: Commit**

```bash
git add backend/schema.sql
git commit -m "feat: add evaluations table, delivery_days, purchase_orders fields"
```

---

## Task 3: Extend SupabaseRepository

**Files:**
- Modify: `backend/src/database/client.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_repository.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/__init__.py` (empty).

Create `backend/tests/test_repository.py`:

```python
from unittest.mock import MagicMock, patch
import pytest
from src.database.client import SupabaseRepository


def _make_repo(data=None):
    """Build a SupabaseRepository with a mocked Supabase client."""
    mock_client = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.return_value.data = [data or {}]
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [data or {}]
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [data or {}]
    repo = SupabaseRepository(client=mock_client)
    return repo, mock_client


def test_create_evaluation_inserts_row():
    repo, mock_client = _make_repo({"session_id": "sess_1", "user_id": "u1", "status": "PLANNING"})
    result = repo.create_evaluation("sess_1", "u1")
    assert result["session_id"] == "sess_1"
    mock_client.table.assert_called_with("evaluations")


def test_update_evaluation_sets_fields():
    repo, mock_client = _make_repo({"session_id": "sess_1", "status": "EXECUTING"})
    repo.update_evaluation("sess_1", status="EXECUTING", current_step="check_stock")
    mock_client.table.assert_called_with("evaluations")


def test_get_evaluation_returns_row():
    repo, mock_client = _make_repo({"session_id": "sess_1", "status": "AWAITING_APPROVAL"})
    result = repo.get_evaluation("sess_1")
    assert result["session_id"] == "sess_1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_repository.py -v
```

Expected: `AttributeError: 'SupabaseRepository' object has no attribute 'create_evaluation'`

- [ ] **Step 3: Add new methods to SupabaseRepository**

Add to the domain methods section in `backend/src/database/client.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_repository.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/src/database/client.py backend/tests/
git commit -m "feat: add evaluation CRUD and purchase_order_full to SupabaseRepository"
```

---

## Task 4: ProcurementState TypedDict

**Files:**
- Create: `backend/src/core/state.py`

- [ ] **Step 1: Create state.py**

Create `backend/src/core/state.py`:

```python
from __future__ import annotations
from typing import Any, TypedDict


class ProcurementState(TypedDict, total=False):
    # Session metadata
    session_id: str
    user_id: str
    user_message: str
    session_start_ts: str       # ISO timestamp, used as Gmail search lower bound

    # Planning
    plan: list[str]
    plan_attempts: int
    plan_error: str | None
    validation_passed: bool

    # Execution
    current_step: str
    status: str                 # PLANNING | EXECUTING | AWAITING_APPROVAL | APPROVED | FAILED
    error: str | None

    # check_stock outputs
    item_name: str
    item_id: str
    requested_qty: int
    stock_sufficient: bool
    current_stock: int

    # send_rfqs outputs
    rfq_sent_at: str            # ISO timestamp
    supplier_emails: list[str]

    # wait_for_quotes outputs
    all_replied: bool

    # extract_quotes outputs
    extracted_quotes: list[dict[str, Any]]
    # each: {supplier_id, supplier_name, unit_price_sen, quoted_delivery_days, payment_terms}

    # query_history outputs
    avg_unit_price_sen: float
    avg_delivery_days: float

    # evaluate_suppliers outputs
    evaluated_suppliers: list[dict[str, Any]]
    # each: {supplier_id, supplier_name, unit_price_sen, quoted_delivery_days, payment_terms,
    #        price_score, delivery_score, payment_terms_score, total_score, risk_flags, is_recommended}

    # generate_report outputs
    report_markdown: str
```

- [ ] **Step 2: Verify import works**

```bash
cd backend && python -c "from src.core.state import ProcurementState; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/core/state.py
git commit -m "feat: add ProcurementState TypedDict"
```

---

## Task 5: SSE Infrastructure

**Files:**
- Create: `backend/src/api/sse.py`
- Create: `backend/tests/test_sse.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_sse.py`:

```python
import asyncio
import pytest
from src.api.sse import create_session, push_event, end_stream, format_sse


@pytest.mark.asyncio
async def test_push_and_drain_events():
    create_session("sess_test")
    await push_event("sess_test", "step_start", {"step": "check_stock"})
    await push_event("sess_test", "step_done", {"step": "check_stock"})
    await end_stream("sess_test")

    from src.api.sse import _queues
    # queue should be removed after end_stream
    assert "sess_test" not in _queues


@pytest.mark.asyncio
async def test_format_sse_output():
    event = {"type": "plan", "data": {"steps": ["check_stock"]}}
    result = format_sse(event)
    assert result.startswith("event: plan\n")
    assert '"steps"' in result


@pytest.mark.asyncio
async def test_push_to_unknown_session_does_nothing():
    # Should not raise
    await push_event("nonexistent", "step_start", {"step": "x"})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && python -m pytest tests/test_sse.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.sse'`

- [ ] **Step 3: Create sse.py**

Create `backend/src/api/sse.py`:

```python
from __future__ import annotations

import asyncio
import json
from typing import Any

_queues: dict[str, asyncio.Queue] = {}


def create_session(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _queues[session_id] = q
    return q


def get_queue(session_id: str) -> asyncio.Queue | None:
    return _queues.get(session_id)


async def push_event(session_id: str, event_type: str, data: dict[str, Any]) -> None:
    q = _queues.get(session_id)
    if q:
        await q.put({"type": event_type, "data": data})


async def end_stream(session_id: str) -> None:
    q = _queues.get(session_id)
    if q:
        await q.put(None)  # sentinel
    _queues.pop(session_id, None)


def format_sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_sse.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/sse.py backend/tests/test_sse.py
git commit -m "feat: add SSE session queue infrastructure"
```

---

## Task 6: Deterministic Scoring Engine

**Files:**
- Create: `backend/src/services/scoring.py`
- Create: `backend/tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_scoring.py`:

```python
import pytest
from src.services.scoring import (
    min_max_invert,
    payment_terms_score,
    score_suppliers,
)


def test_min_max_invert_lower_is_better():
    # values = [410000, 395000]; lower price → higher score
    assert min_max_invert([410000, 395000], 395000) == pytest.approx(100.0)
    assert min_max_invert([410000, 395000], 410000) == pytest.approx(0.0)


def test_min_max_invert_all_same():
    # Avoid division by zero — return 100
    assert min_max_invert([400000, 400000], 400000) == pytest.approx(100.0)


def test_payment_terms_score_known_values():
    assert payment_terms_score("Net-60") == 100
    assert payment_terms_score("Net-45") == 75
    assert payment_terms_score("Net-30") == 50
    assert payment_terms_score("Net-15") == 25
    assert payment_terms_score("Immediate") == 0
    assert payment_terms_score("Unknown terms") == 50  # default


def test_payment_terms_score_case_insensitive():
    assert payment_terms_score("net-60") == 100
    assert payment_terms_score("NET-30") == 50


def test_score_suppliers_ranking():
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 410000,
            "quoted_delivery_days": 5,
            "payment_terms": "Net-30",
        },
        {
            "supplier_id": "SUP-B",
            "supplier_name": "Global IT",
            "unit_price_sen": 395000,
            "quoted_delivery_days": 2,
            "payment_terms": "Net-60",
        },
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)

    sup_b = next(r for r in results if r["supplier_id"] == "SUP-B")
    sup_a = next(r for r in results if r["supplier_id"] == "SUP-A")

    assert sup_b["total_score"] > sup_a["total_score"]
    assert sup_b["is_recommended"] is True
    assert sup_a["is_recommended"] is False


def test_score_suppliers_risk_flag_price():
    # SUP-A price 410000, avg 365000 → 410000/365000 = 1.123 → >10% → flag
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 410000,
            "quoted_delivery_days": 5,
            "payment_terms": "Net-30",
        }
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)
    flags = results[0]["risk_flags"]
    assert any("Price" in f and "above historical" in f for f in flags)


def test_score_suppliers_no_risk_flag_below_threshold():
    # Price exactly at 1.05x avg — below 10% threshold
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 383250,  # 365000 * 1.05
            "quoted_delivery_days": 7,
            "payment_terms": "Net-30",
        }
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)
    assert results[0]["risk_flags"] == []


def test_score_suppliers_risk_flag_delivery():
    # Delivery 12 days, avg 7 → 12/7 = 1.71 → >10% → flag
    quotes = [
        {
            "supplier_id": "SUP-A",
            "supplier_name": "Alpha Tech",
            "unit_price_sen": 365000,
            "quoted_delivery_days": 12,
            "payment_terms": "Net-30",
        }
    ]
    results = score_suppliers(quotes, avg_historical_price_sen=365000, avg_historical_delivery_days=7.0)
    flags = results[0]["risk_flags"]
    assert any("Delivery" in f and "above historical" in f for f in flags)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && python -m pytest tests/test_scoring.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.services'`

- [ ] **Step 3: Create scoring.py**

Create `backend/src/services/__init__.py` (empty).

Create `backend/src/services/scoring.py`:

```python
from __future__ import annotations

PAYMENT_TERMS_SCORES: dict[str, int] = {
    "net-60": 100,
    "net-45": 75,
    "net-30": 50,
    "net-15": 25,
    "immediate": 0,
}


def min_max_invert(values: list[float], value: float) -> float:
    """Normalize value where lower is better → higher score."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return 100.0
    return (hi - value) / (hi - lo) * 100.0


def payment_terms_score(terms: str) -> int:
    normalized = terms.lower().strip()
    for key, score in PAYMENT_TERMS_SCORES.items():
        if key in normalized:
            return score
    return 50  # unknown


def score_suppliers(
    quotes: list[dict],
    avg_historical_price_sen: float,
    avg_historical_delivery_days: float,
) -> list[dict]:
    """
    Score each supplier. Returns enriched list with score fields + risk_flags + is_recommended.
    Weights: price 55%, delivery 30%, payment_terms 15%.
    """
    prices = [float(q["unit_price_sen"]) for q in quotes]
    deliveries = [float(q["quoted_delivery_days"]) for q in quotes]

    results = []
    for q in quotes:
        p_score = min_max_invert(prices, float(q["unit_price_sen"]))
        d_score = min_max_invert(deliveries, float(q["quoted_delivery_days"]))
        pt_score = float(payment_terms_score(q.get("payment_terms", "Unknown")))
        total = p_score * 0.55 + d_score * 0.30 + pt_score * 0.15

        risk_flags: list[str] = []
        if avg_historical_price_sen > 0:
            ratio = q["unit_price_sen"] / avg_historical_price_sen
            if ratio > 1.10:
                pct = (ratio - 1.0) * 100
                risk_flags.append(f"Price {pct:.0f}% above historical average")
        if avg_historical_delivery_days > 0:
            ratio = q["quoted_delivery_days"] / avg_historical_delivery_days
            if ratio > 1.10:
                pct = (ratio - 1.0) * 100
                risk_flags.append(f"Delivery {pct:.0f}% above historical average")

        results.append({
            **q,
            "price_score": round(p_score, 2),
            "delivery_score": round(d_score, 2),
            "payment_terms_score": round(pt_score, 2),
            "total_score": round(total, 2),
            "risk_flags": risk_flags,
            "is_recommended": False,  # set below
        })

    best = max(results, key=lambda r: r["total_score"])
    for r in results:
        r["is_recommended"] = r["supplier_id"] == best["supplier_id"]

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_scoring.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/ backend/tests/test_scoring.py
git commit -m "feat: add deterministic scoring engine with risk flags"
```

---

## Task 7: Gmail Service

**Files:**
- Create: `backend/src/services/gmail.py`

- [ ] **Step 1: Create gmail.py**

Create `backend/src/services/gmail.py`:

```python
from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]


def get_gmail_service(credentials_path: str, token_path: str):
    """Build and return an authenticated Gmail API service object."""
    creds: Credentials | None = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(service, to: str, subject: str, body_html: str) -> str:
    """Send an email. Returns the Gmail message ID."""
    msg = MIMEText(body_html, "html")
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return result["id"]


def fetch_replies(
    service,
    from_emails: list[str],
    since_timestamp: str,
) -> list[dict[str, Any]]:
    """
    Fetch reply emails from each address in from_emails sent after since_timestamp.

    since_timestamp: ISO 8601 string e.g. '2026-07-01T10:00:00Z'
    Returns list of:
        {
            "from": str,
            "subject": str,
            "body_text": str,
            "attachments": [{"filename": str, "data": bytes}]
        }
    """
    import email as email_lib
    from datetime import datetime, timezone

    # Convert ISO timestamp to Unix epoch for Gmail query
    dt = datetime.fromisoformat(since_timestamp.replace("Z", "+00:00"))
    epoch = int(dt.timestamp())

    replies = []
    for sender_email in from_emails:
        query = f"from:{sender_email} after:{epoch}"
        try:
            result = service.users().messages().list(userId="me", q=query).execute()
        except HttpError:
            continue

        messages = result.get("messages", [])
        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="full"
            ).execute()

            headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
            body_text = _extract_body(msg["payload"])
            attachments = _extract_attachments(service, msg)

            replies.append({
                "from": headers.get("From", sender_email),
                "subject": headers.get("Subject", ""),
                "body_text": body_text,
                "attachments": attachments,
            })

    return replies


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text body from Gmail message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


def _extract_attachments(service, msg: dict) -> list[dict[str, Any]]:
    """Download all PDF attachments from a Gmail message."""
    attachments = []
    payload = msg["payload"]

    for part in payload.get("parts", []):
        filename = part.get("filename", "")
        if not filename.lower().endswith(".pdf"):
            continue
        attachment_id = part["body"].get("attachmentId")
        if not attachment_id:
            continue
        att = service.users().messages().attachments().get(
            userId="me", messageId=msg["id"], id=attachment_id
        ).execute()
        data = base64.urlsafe_b64decode(att["data"] + "==")
        attachments.append({"filename": filename, "data": data})

    return attachments
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.services.gmail import get_gmail_service, send_email, fetch_replies; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/services/gmail.py
git commit -m "feat: add Gmail service (send + fetch replies with attachments)"
```

---

## Task 8: FastMCP Server Scaffold

**Files:**
- Create: `backend/src/mcp_server.py`

- [ ] **Step 1: Create mcp_server.py**

Create `backend/src/mcp_server.py`:

```python
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
```

- [ ] **Step 2: Verify import + validate_plan**

```bash
cd backend && python -c "
from src.mcp_server import validate_plan
assert validate_plan(['check_stock', 'generate_report']) is None
assert validate_plan(['unknown_step']) is not None
assert validate_plan(['evaluate_suppliers', 'extract_quotes']) is not None
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/mcp_server.py
git commit -m "feat: add FastMCP server scaffold with plan validation"
```

---

## Task 9: check_stock Tool

**Files:**
- Create: `backend/src/agents/tools/__init__.py`
- Create: `backend/src/agents/tools/stock.py`

- [ ] **Step 1: Create tool files**

Create `backend/src/agents/tools/__init__.py` (empty).

Create `backend/src/agents/tools/stock.py`:

```python
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
    result = await check_stock(state["item_name"], state["requested_qty"])
    return {
        **state,
        "item_id": result["item_id"],
        "current_stock": result["current_stock"],
        "stock_sufficient": result["stock_sufficient"],
    }
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.stock import check_stock_handler; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/
git commit -m "feat: add check_stock MCP tool and handler"
```

---

## Task 10: send_rfqs Tool

**Files:**
- Create: `backend/src/agents/tools/rfq.py`

- [ ] **Step 1: Create rfq.py**

Create `backend/src/agents/tools/rfq.py`:

```python
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
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
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.rfq import send_rfqs_handler; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/rfq.py
git commit -m "feat: add send_rfqs MCP tool (Gemini RFQ draft + Gmail send)"
```

---

## Task 11: wait_for_quotes + extract_quotes Tools

**Files:**
- Create: `backend/src/agents/tools/quotes.py`

- [ ] **Step 1: Create quotes.py**

Create `backend/src/agents/tools/quotes.py`:

```python
from __future__ import annotations

import asyncio
import io
from typing import Any

import pypdf
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.state import ProcurementState
from src.mcp_server import mcp
from src.services.gmail import fetch_replies, get_gmail_service

_POLL_INTERVAL = 15  # seconds
_DEFAULT_TIMEOUT = 300  # 5 minutes


@mcp.tool(description="Poll Gmail every 15s for supplier replies; proceed when all replied or 5-min timeout")
async def wait_for_quotes(
    supplier_emails: list[str],
    rfq_sent_at: str,
    timeout_seconds: int = _DEFAULT_TIMEOUT,
) -> dict:
    """MCP tool: block until all suppliers reply or timeout."""
    settings = get_settings()
    service = get_gmail_service(settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH)

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        replies = fetch_replies(service, from_emails=supplier_emails, since_timestamp=rfq_sent_at)
        replied_from = {r["from"].split("<")[-1].strip(">").strip() for r in replies}
        if all(email in replied_from for email in supplier_emails):
            return {"all_replied": True}
        await asyncio.sleep(_POLL_INTERVAL)

    return {"all_replied": False}


async def wait_for_quotes_handler(state: ProcurementState) -> ProcurementState:
    result = await wait_for_quotes(
        supplier_emails=state["supplier_emails"],
        rfq_sent_at=state["rfq_sent_at"],
    )
    return {**state, "all_replied": result["all_replied"]}


class _QuoteExtracted(BaseModel):
    supplier_name: str
    unit_price_sen: int
    quoted_delivery_days: int
    payment_terms: str


class _QuotesOutput(BaseModel):
    quotes: list[_QuoteExtracted]


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_quotes_with_gemini(text: str, api_key: str) -> list[dict[str, Any]]:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    structured = llm.with_structured_output(_QuotesOutput)
    prompt = (
        "Extract procurement quote data from this supplier communication.\n\n"
        f"Text:\n{text}\n\n"
        "Extract all quotes found. For each quote:\n"
        "- supplier_name: company name\n"
        "- unit_price_sen: unit price in sen (Malaysian Ringgit cents; RM 1 = 100 sen; e.g. RM 3,950.00 = 395000)\n"
        "- quoted_delivery_days: delivery time as integer number of days\n"
        "- payment_terms: e.g. 'Net-30', 'Net-60', 'Net-45', 'Net-15', 'Immediate'"
    )
    result = structured.invoke(prompt)
    return [q.model_dump() for q in result.quotes]


@mcp.tool(description="Fetch reply emails + PDF attachments via Gmail; extract structured quote data with Gemini")
async def extract_quotes(supplier_emails: list[str], rfq_sent_at: str) -> dict:
    """MCP tool: read Gmail replies, parse PDFs/bodies, extract structured quotes."""
    settings = get_settings()
    service = get_gmail_service(settings.GMAIL_CREDENTIALS_PATH, settings.GMAIL_TOKEN_PATH)

    replies = fetch_replies(service, from_emails=supplier_emails, since_timestamp=rfq_sent_at)
    if not replies:
        raise ValueError("No supplier replies found in Gmail")

    db_suppliers = None  # lazy-load only if needed for supplier_id lookup

    all_quotes: list[dict[str, Any]] = []
    for reply in replies:
        # Prefer PDF attachment text, fall back to email body
        if reply["attachments"]:
            text = _extract_text_from_pdf(reply["attachments"][0]["data"])
        else:
            text = reply["body_text"]

        if not text.strip():
            continue

        quotes = _parse_quotes_with_gemini(text, settings.GOOGLE_API_KEY)
        # Enrich each quote with supplier_id by matching supplier_name to DB
        if db_suppliers is None:
            from src.database.client import SupabaseRepository
            db = SupabaseRepository()
            db_suppliers = db.get_all_suppliers()

        for q in quotes:
            matched = next(
                (s for s in db_suppliers if s["name"].lower() in q["supplier_name"].lower()),
                None,
            )
            q["supplier_id"] = matched["supplier_id"] if matched else "UNKNOWN"
            all_quotes.append(q)

    return {"extracted_quotes": all_quotes}


async def extract_quotes_handler(state: ProcurementState) -> ProcurementState:
    result = await extract_quotes(
        supplier_emails=state["supplier_emails"],
        rfq_sent_at=state["rfq_sent_at"],
    )
    return {**state, "extracted_quotes": result["extracted_quotes"]}
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.quotes import wait_for_quotes_handler, extract_quotes_handler; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/quotes.py
git commit -m "feat: add wait_for_quotes and extract_quotes MCP tools"
```

---

## Task 12: query_history Tool

**Files:**
- Create: `backend/src/agents/tools/history.py`

- [ ] **Step 1: Create history.py**

Create `backend/src/agents/tools/history.py`:

```python
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
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.history import query_history_handler; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/history.py
git commit -m "feat: add query_history MCP tool (historical avg price + delivery)"
```

---

## Task 13: evaluate_suppliers Tool

**Files:**
- Create: `backend/src/agents/tools/evaluation.py`

- [ ] **Step 1: Create evaluation.py**

Create `backend/src/agents/tools/evaluation.py`:

```python
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
    result = await evaluate_suppliers(
        extracted_quotes=state.get("extracted_quotes", []),
        avg_unit_price_sen=state.get("avg_unit_price_sen", 0.0),
        avg_delivery_days=state.get("avg_delivery_days", 0.0),
    )
    return {**state, "evaluated_suppliers": result["evaluated_suppliers"]}
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.evaluation import evaluate_suppliers_handler; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/evaluation.py
git commit -m "feat: add evaluate_suppliers MCP tool"
```

---

## Task 14: generate_report Tool

**Files:**
- Create: `backend/src/agents/tools/report.py`

- [ ] **Step 1: Create report.py**

Create `backend/src/agents/tools/report.py`:

```python
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
    result = await generate_report(
        evaluated_suppliers=state.get("evaluated_suppliers", []),
        item_name=state.get("item_name", ""),
        requested_qty=state.get("requested_qty", 0),
        stock_sufficient=state.get("stock_sufficient"),
        current_stock=state.get("current_stock"),
    )
    return {**state, "report_markdown": result["report_markdown"]}
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.report import generate_report_handler; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/report.py
git commit -m "feat: add generate_report MCP tool (markdown assembly, no LLM)"
```

---

## Task 15: Automation Tool (Post-Approval)

**Files:**
- Create: `backend/src/agents/tools/automation.py`

- [ ] **Step 1: Create automation.py**

Create `backend/src/agents/tools/automation.py`:

```python
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
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.agents.tools.automation import run_automation; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/tools/automation.py
git commit -m "feat: add automation agent (PO write + PDF generation + Supabase Storage)"
```

---

## Task 16: LangGraph Manager Agent

**Files:**
- Create: `backend/src/agents/manager.py`

- [ ] **Step 1: Create manager.py**

Create `backend/src/agents/manager.py`:

```python
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.agents.tools.evaluation import evaluate_suppliers_handler
from src.agents.tools.history import query_history_handler
from src.agents.tools.quotes import extract_quotes_handler, wait_for_quotes_handler
from src.agents.tools.report import generate_report_handler
from src.agents.tools.rfq import send_rfqs_handler
from src.agents.tools.stock import check_stock_handler
from src.api.sse import end_stream, push_event
from src.core.config import get_settings
from src.core.state import ProcurementState
from src.database.client import SupabaseRepository
from src.mcp_server import TOOL_DESCRIPTIONS, validate_plan

# ── Tool registry: step name → async handler function ─────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "check_stock": check_stock_handler,
    "send_rfqs": send_rfqs_handler,
    "wait_for_quotes": wait_for_quotes_handler,
    "extract_quotes": extract_quotes_handler,
    "query_history": query_history_handler,
    "evaluate_suppliers": evaluate_suppliers_handler,
    "generate_report": generate_report_handler,
}

STEP_MESSAGES: dict[str, str] = {
    "check_stock": "Checking inventory stock levels...",
    "send_rfqs": "Drafting and sending RFQ emails to suppliers...",
    "wait_for_quotes": "Waiting for supplier replies (checking every 15 seconds)...",
    "extract_quotes": "Reading supplier replies and extracting quote data...",
    "query_history": "Querying historical purchase data...",
    "evaluate_suppliers": "Evaluating and scoring suppliers...",
    "generate_report": "Assembling recommendation report...",
}


# ── Pydantic schema for Gemini structured output ───────────────────────────

class _PlanOutput(BaseModel):
    plan: list[str]


# ── Graph nodes ────────────────────────────────────────────────────────────

async def plan_node(state: ProcurementState) -> ProcurementState:
    """Call Gemini with structured output to generate a step-by-step plan."""
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GOOGLE_API_KEY)
    structured = llm.with_structured_output(_PlanOutput)

    tool_list = "\n".join(
        f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
    )
    error_context = ""
    if state.get("plan_error"):
        error_context = f"\n\nPrevious plan was invalid: {state['plan_error']}\nPlease fix the ordering and try again."

    prompt = (
        f"You are a procurement orchestrator. Given the user's request, choose the minimum "
        f"set of steps needed and return them as an ordered plan.\n\n"
        f"Available steps:\n{tool_list}\n\n"
        f"Ordering rules:\n"
        f"- send_rfqs must come before wait_for_quotes\n"
        f"- wait_for_quotes must come before extract_quotes\n"
        f"- extract_quotes must come before evaluate_suppliers\n"
        f"- query_history must come before evaluate_suppliers\n"
        f"- evaluate_suppliers must come before generate_report\n"
        f"{error_context}\n\n"
        f"User request: {state['user_message']}"
    )

    result = await structured.ainvoke([HumanMessage(content=prompt)])
    attempts = state.get("plan_attempts", 0) + 1
    return {**state, "plan": result.plan, "plan_attempts": attempts, "plan_error": None}


async def validate_node(state: ProcurementState) -> ProcurementState:
    """Validate the generated plan against registry + ordering rules."""
    error = validate_plan(state.get("plan", []))
    return {**state, "plan_error": error, "validation_passed": error is None}


def _route_after_validate(state: ProcurementState) -> str:
    if state.get("validation_passed"):
        return "stream_plan"
    if state.get("plan_attempts", 0) < 2:
        return "plan"
    return "error"


async def stream_plan_node(state: ProcurementState) -> ProcurementState:
    """Push the validated plan to the user via SSE."""
    plan = state.get("plan", [])
    readable = " → ".join(step.replace("_", " ") for step in plan)
    await push_event(
        state["session_id"],
        "plan",
        {"steps": plan, "message": f"My plan: {readable}"},
    )
    return {**state, "status": "EXECUTING"}


async def execute_node(state: ProcurementState) -> ProcurementState:
    """Execute each step in the plan, streaming status, persisting state after each step."""
    db = SupabaseRepository()
    session_id = state["session_id"]

    for step in state.get("plan", []):
        await push_event(session_id, "step_start", {"step": step, "message": STEP_MESSAGES.get(step, step)})
        try:
            state = await TOOL_REGISTRY[step](state)
        except Exception as exc:
            error_msg = f"Step '{step}' failed: {exc}"
            await push_event(session_id, "error", {"step": step, "message": error_msg})
            await end_stream(session_id)
            db.update_evaluation(session_id, status="FAILED", current_step=step, state_json=dict(state))
            return {**state, "status": "FAILED", "error": error_msg, "current_step": step}

        state["current_step"] = step
        db.update_evaluation(session_id, current_step=step, state_json=dict(state))
        await push_event(session_id, "step_done", {"step": step, "message": f"✓ Completed"})

    await push_event(session_id, "report", {"markdown": state.get("report_markdown", "")})
    await push_event(session_id, "approve_ready", {"session_id": session_id, "message": "Approve to generate purchase order"})
    await end_stream(session_id)

    state["status"] = "AWAITING_APPROVAL"
    db.update_evaluation(
        session_id,
        status="AWAITING_APPROVAL",
        report_markdown=state.get("report_markdown", ""),
        state_json=dict(state),
    )
    return state


async def error_node(state: ProcurementState) -> ProcurementState:
    """Stream planning error to user and mark session as FAILED."""
    db = SupabaseRepository()
    session_id = state["session_id"]
    error_msg = state.get("plan_error", "Failed to generate a valid plan after 2 attempts")
    await push_event(session_id, "error", {"step": "planning", "message": error_msg})
    await end_stream(session_id)
    db.update_evaluation(session_id, status="FAILED")
    return {**state, "status": "FAILED"}


# ── Graph construction ──────────────────────────────────────────────────────

def build_manager_graph():
    graph = StateGraph(ProcurementState)

    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("stream_plan", stream_plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("error", error_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"stream_plan": "stream_plan", "plan": "plan", "error": "error"},
    )
    graph.add_edge("stream_plan", "execute")
    graph.add_edge("execute", END)
    graph.add_edge("error", END)

    return graph.compile()


manager_graph = build_manager_graph()
```

- [ ] **Step 2: Verify import and graph compiles**

```bash
cd backend && python -c "from src.agents.manager import manager_graph; print('Graph compiled OK')"
```

Expected: `Graph compiled OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/agents/manager.py
git commit -m "feat: add LangGraph plan-then-execute Manager Agent"
```

---

## Task 17: FastAPI Chat Routes

**Files:**
- Modify: `backend/src/api/routes.py` (full replacement)

- [ ] **Step 1: Replace routes.py**

Replace the entire content of `backend/src/api/routes.py`:

```python
"""
api/routes.py — FastAPI chat endpoints for the agentic procurement system.

POST  /api/v1/chat                       — start a procurement session
GET   /api/v1/chat/{session_id}/stream   — SSE stream of agent updates
POST  /api/v1/chat/{session_id}/approve  — trigger Automation Agent post-approval
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.manager import manager_graph
from src.agents.tools.automation import run_automation
from src.api.sse import create_session, end_stream, get_queue, push_event
from src.database.client import SupabaseRepository

router = APIRouter(prefix="/api/v1", tags=["chat"])

db = SupabaseRepository()


# ── Request / Response models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"


class ChatCreated(BaseModel):
    session_id: str


class ApproveResponse(BaseModel):
    status: str
    po_pdf_url: str
    po_number: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatCreated, status_code=202)
async def create_chat(body: ChatRequest) -> ChatCreated:
    """
    Start a new procurement session. Returns session_id immediately.
    Connect to /chat/{session_id}/stream to receive SSE updates.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session_start_ts = datetime.now(timezone.utc).isoformat()

    # Parse item_name and requested_qty from the message
    # The Manager Agent will do the deep parsing — we seed minimal state here
    initial_state = {
        "session_id": session_id,
        "user_id": body.user_id,
        "user_message": body.message,
        "session_start_ts": session_start_ts,
        "plan_attempts": 0,
        "item_name": _extract_item_name(body.message),
        "requested_qty": _extract_quantity(body.message),
    }

    # Create SSE queue and evaluation row before starting background task
    create_session(session_id)
    db.create_evaluation(session_id, body.user_id)

    asyncio.create_task(_run_pipeline(session_id, initial_state))
    return ChatCreated(session_id=session_id)


@router.get("/chat/{session_id}/stream")
async def stream_chat(session_id: str):
    """SSE stream for a procurement session. Connect immediately after POST /chat."""
    q = get_queue(session_id)
    if q is None:
        # Session may have already completed — check DB
        record = db.get_evaluation(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # Stream a synthetic done event
        async def _replay():
            if record.get("report_markdown"):
                yield {"event": "report", "data": json.dumps({"markdown": record["report_markdown"]})}
                yield {"event": "approve_ready", "data": json.dumps({"session_id": session_id})}
        return EventSourceResponse(_replay())

    async def _generator():
        while True:
            event = await q.get()
            if event is None:
                break
            yield {"event": event["type"], "data": json.dumps(event["data"])}

    return EventSourceResponse(_generator())


@router.post("/chat/{session_id}/approve", response_model=ApproveResponse)
async def approve_chat(session_id: str) -> ApproveResponse:
    """Trigger Automation Agent: generate PO, PDF, upload to Supabase Storage."""
    record = db.get_evaluation(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if record["status"] != "AWAITING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Session status is '{record['status']}', expected 'AWAITING_APPROVAL'",
        )

    state = record.get("state_json") or {}
    try:
        updated_state = await run_automation(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Automation failed: {exc}") from exc

    db.update_evaluation(session_id, status="APPROVED")

    return ApproveResponse(
        status="SUCCESS",
        po_pdf_url=updated_state.get("po_pdf_url", ""),
        po_number=updated_state.get("po_number", ""),
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

async def _run_pipeline(session_id: str, initial_state: dict) -> None:
    """Run the LangGraph manager graph as a background task."""
    try:
        await manager_graph.ainvoke(initial_state)
    except Exception as exc:
        await push_event(session_id, "error", {"step": "pipeline", "message": str(exc)})
        await end_stream(session_id)
        db.update_evaluation(session_id, status="FAILED")


def _extract_item_name(message: str) -> str:
    """
    Naive item name extraction from the user message.
    The Manager Agent planning prompt handles proper interpretation,
    but we need a seed value for the state.
    """
    # Strip leading quantity words, return rest as item name
    words = message.split()
    for i, word in enumerate(words):
        if word.isdigit():
            return " ".join(words[i + 1:]) if i + 1 < len(words) else message
    return message


def _extract_quantity(message: str) -> int:
    """Extract first integer from message as requested quantity."""
    for word in message.split():
        cleaned = word.rstrip(".,")
        if cleaned.isdigit():
            return int(cleaned)
    return 1
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from src.api.routes import router; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/src/api/routes.py
git commit -m "feat: replace evaluation endpoints with SSE chat + approve routes"
```

---

## Task 18: Wire Up main.py

**Files:**
- Modify: `backend/src/main.py`

- [ ] **Step 1: Update main.py**

Replace the contents of `backend/src/main.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as chat_router

# Import mcp_server to trigger @mcp.tool() registrations in all tool modules
import src.mcp_server  # noqa: F401
import src.agents.tools.stock  # noqa: F401
import src.agents.tools.rfq  # noqa: F401
import src.agents.tools.quotes  # noqa: F401
import src.agents.tools.history  # noqa: F401
import src.agents.tools.evaluation  # noqa: F401
import src.agents.tools.report  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from src.core.config import get_settings
    get_settings()  # raises early if SUPABASE_URL / KEY missing
    yield


app = FastAPI(
    title="AI Procurement Operations API",
    description="Multi-Agent System backend. Powered by LangGraph + Gemini + FastMCP.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": "ai-procurement-ops", "status": "healthy", "version": "0.2.0"}


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 2: Start server and verify it starts**

```bash
cd backend && uvicorn src.main:app --reload --port 8000
```

Expected: `Application startup complete` with no import errors. Visit `http://localhost:8000/docs` — should show `/api/v1/chat` endpoints.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main.py
git commit -m "feat: wire up chat router and MCP tool registrations in main.py"
```

---

## Task 19: Tool Unit Tests

**Files:**
- Create: `backend/tests/test_tools.py`

- [ ] **Step 1: Create test_tools.py**

Create `backend/tests/test_tools.py`:

```python
"""Unit tests for tool handlers using mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.core.state import ProcurementState


# ── check_stock ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_stock_sufficient():
    mock_db = MagicMock()
    mock_db.select.return_value = [
        {"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "current_stock": 50}
    ]
    with patch("src.agents.tools.stock.SupabaseRepository", return_value=mock_db):
        from src.agents.tools.stock import check_stock_handler
        state: ProcurementState = {
            "session_id": "s1",
            "user_message": "Buy 30 laptops",
            "item_name": "Dell XPS 15 Laptop",
            "requested_qty": 30,
        }
        result = await check_stock_handler(state)
        assert result["stock_sufficient"] is True
        assert result["item_id"] == "IT-XPS-15"
        assert result["current_stock"] == 50


@pytest.mark.asyncio
async def test_check_stock_insufficient():
    mock_db = MagicMock()
    mock_db.select.return_value = [
        {"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "current_stock": 4}
    ]
    with patch("src.agents.tools.stock.SupabaseRepository", return_value=mock_db):
        from src.agents.tools.stock import check_stock_handler
        state: ProcurementState = {
            "session_id": "s1",
            "user_message": "Buy 30 laptops",
            "item_name": "Dell XPS 15 Laptop",
            "requested_qty": 30,
        }
        result = await check_stock_handler(state)
        assert result["stock_sufficient"] is False


# ── query_history ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_history_computes_averages():
    mock_db = MagicMock()
    mock_db.get_purchase_history.return_value = [
        {"unit_price_sen": 365000, "delivery_days": 7},
        {"unit_price_sen": 385000, "delivery_days": 9},
    ]
    with patch("src.agents.tools.history.SupabaseRepository", return_value=mock_db):
        from src.agents.tools.history import query_history_handler
        state: ProcurementState = {
            "session_id": "s1",
            "user_message": "Buy laptops",
            "item_id": "IT-XPS-15",
        }
        result = await query_history_handler(state)
        assert result["avg_unit_price_sen"] == pytest.approx(375000.0)
        assert result["avg_delivery_days"] == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_query_history_no_history_returns_zeros():
    mock_db = MagicMock()
    mock_db.get_purchase_history.return_value = []
    with patch("src.agents.tools.history.SupabaseRepository", return_value=mock_db):
        from src.agents.tools.history import query_history_handler
        state: ProcurementState = {"session_id": "s1", "user_message": "Buy laptops", "item_id": "IT-NEW"}
        result = await query_history_handler(state)
        assert result["avg_unit_price_sen"] == 0.0
        assert result["avg_delivery_days"] == 0.0


# ── evaluate_suppliers ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_suppliers_handler():
    from src.agents.tools.evaluation import evaluate_suppliers_handler
    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "Buy laptops",
        "extracted_quotes": [
            {"supplier_id": "SUP-A", "supplier_name": "Alpha Tech", "unit_price_sen": 410000, "quoted_delivery_days": 5, "payment_terms": "Net-30"},
            {"supplier_id": "SUP-B", "supplier_name": "Global IT", "unit_price_sen": 395000, "quoted_delivery_days": 2, "payment_terms": "Net-60"},
        ],
        "avg_unit_price_sen": 365000.0,
        "avg_delivery_days": 7.0,
    }
    result = await evaluate_suppliers_handler(state)
    evaluated = result["evaluated_suppliers"]
    assert len(evaluated) == 2
    recommended = next(s for s in evaluated if s["is_recommended"])
    assert recommended["supplier_id"] == "SUP-B"


# ── generate_report ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_report_handler_contains_key_fields():
    from src.agents.tools.report import generate_report_handler
    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "Buy 30 laptops",
        "item_name": "Dell XPS 15 Laptop",
        "requested_qty": 30,
        "stock_sufficient": False,
        "current_stock": 4,
        "evaluated_suppliers": [
            {
                "supplier_id": "SUP-B",
                "supplier_name": "Global IT",
                "unit_price_sen": 395000,
                "quoted_delivery_days": 2,
                "payment_terms": "Net-60",
                "price_score": 100.0,
                "delivery_score": 100.0,
                "payment_terms_score": 100.0,
                "total_score": 100.0,
                "risk_flags": [],
                "is_recommended": True,
            }
        ],
    }
    result = await generate_report_handler(state)
    md = result["report_markdown"]
    assert "Global IT" in md
    assert "Dell XPS 15" in md
    assert "Recommended" in md or "recommended" in md
```

- [ ] **Step 2: Run tool tests**

```bash
cd backend && python -m pytest tests/test_tools.py -v
```

Expected: `7 passed` (some may be skipped if mocking is complex — 0 failures required)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_tools.py
git commit -m "test: add tool unit tests (check_stock, query_history, evaluate, report)"
```

---

## Task 20: End-to-End Smoke Test

**Files:**
- Create: `backend/tests/test_smoke.py`

- [ ] **Step 1: Create smoke test**

Create `backend/tests/test_smoke.py`:

```python
"""
Smoke test: verify the graph compiles, plan validation works, and routes import.
Does NOT make real Gemini/Gmail/Supabase calls.
"""

import pytest
from src.mcp_server import validate_plan
from src.agents.manager import manager_graph, TOOL_REGISTRY


def test_all_registry_tools_present():
    expected = {"check_stock", "send_rfqs", "wait_for_quotes", "extract_quotes",
                "query_history", "evaluate_suppliers", "generate_report"}
    assert set(TOOL_REGISTRY.keys()) == expected


def test_validate_plan_full_valid():
    plan = ["check_stock", "send_rfqs", "wait_for_quotes", "extract_quotes",
            "query_history", "evaluate_suppliers", "generate_report"]
    assert validate_plan(plan) is None


def test_validate_plan_stock_only():
    assert validate_plan(["check_stock", "generate_report"]) is None


def test_validate_plan_rejects_unknown():
    assert validate_plan(["check_stock", "hack_the_planet"]) is not None


def test_validate_plan_rejects_bad_order():
    # evaluate before extract — invalid
    assert validate_plan(["evaluate_suppliers", "extract_quotes", "generate_report"]) is not None


def test_manager_graph_compiled():
    assert manager_graph is not None


def test_routes_importable():
    from src.api.routes import router
    route_paths = [r.path for r in router.routes]
    assert "/api/v1/chat" in route_paths
    assert any("/stream" in p for p in route_paths)
    assert any("/approve" in p for p in route_paths)
```

- [ ] **Step 2: Run smoke tests**

```bash
cd backend && python -m pytest tests/test_smoke.py -v
```

Expected: `7 passed`

- [ ] **Step 3: Run all tests**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests pass. No failures.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_smoke.py
git commit -m "test: add end-to-end smoke tests for graph, registry, and routes"
```

---

## Self-Review Against Spec

| Spec requirement | Task |
|------------------|------|
| Plan-then-execute Manager Agent (Gemini structured output) | Task 16 |
| Tool registry (7 steps) | Tasks 9–15, Task 16 |
| Plan validation (registry + ordering rules) | Task 8 `validate_plan` |
| Show plan to user before executing (SSE `plan` event) | Task 5, Task 16 `stream_plan_node` |
| Re-prompt once on invalid plan | Task 16 `_route_after_validate` |
| Abort and inform on step failure | Task 16 `execute_node` error handler |
| SSE chat interface | Tasks 5, 17 |
| Gmail outbound RFQs | Task 7, Task 10 |
| Gmail inbound polling (15s, 5min timeout) | Task 11 `wait_for_quotes` |
| pypdf + Gemini structured extraction | Task 11 `extract_quotes` |
| Deterministic scoring (Price 55%, Delivery 30%, PT 15%) | Task 6 |
| Risk flags (>10% above historical) | Task 6 |
| Evaluation persisted to Supabase | Tasks 2, 3, 16 |
| Automation Agent (PO + PDF + Storage) outside plan | Task 15 |
| Approve endpoint triggers automation | Task 17 |
| FastMCP tool registration | Tasks 8–14 |
| `evaluations` table | Task 2 |
| `purchase_orders.item_id/quantity/pdf_url` | Task 2 |
| `purchase_history.delivery_days` | Task 2 |
