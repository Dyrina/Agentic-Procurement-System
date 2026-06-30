# AI Procurement Operations Multi-Agent System — Design Spec

Date: 2026-07-01  
Status: Approved

---

## Overview

Multi-agent agentic procurement system. A Procurement Manager Agent dynamically plans and orchestrates specialized worker tools to automate the full procurement cycle — from RFQ to purchase order — with human approval before commitment.

Architecture: **plan-then-execute orchestrator-worker**, FastMCP in-process, SSE chat interface.

---

## Section 1 — Overall Architecture

```
User
 │  POST /api/v1/chat               (send message)
 │  GET  /api/v1/chat/{id}/stream   (SSE)
 ▼
FastAPI
 │
 ▼
LangGraph Manager Agent
 ├── plan_node       → Gemini structured output → {"plan": [...]}
 ├── validate_node   → registry check + ordering rules (pure Python)
 ├── stream_plan_node → SSE: show plan to user
 └── execute_node    → iterate plan, call each MCP tool, SSE status per step
          │
          ▼ (in-process MCP calls via FastMCP)
     ┌──────────────────────────────────────┐
     │  FastMCP Server (in-process)         │
     │  ├── check_stock                     │
     │  ├── send_rfqs      → Gmail API      │
     │  ├── wait_for_quotes → Gmail poll    │
     │  ├── extract_quotes → pypdf + Gemini │
     │  ├── query_history  → Supabase       │
     │  ├── evaluate_suppliers (scoring)    │
     │  └── generate_report                 │
     └──────────────────────────────────────┘
          │
          ▼ (on completion)
     SSE: report markdown + approve_ready event
          │
     [User clicks Approve]
          │
          ▼
     POST /api/v1/chat/{id}/approve
          │
          ▼
     Automation Agent (hardcoded, outside LLM plan)
     ├── Write PO to Supabase purchase_orders
     ├── Generate PO PDF
     ├── Upload to Supabase Storage
     └── SSE: download link
```

**File structure:**

```
backend/src/
├── agents/
│   ├── manager.py          # LangGraph plan-then-execute graph
│   └── tools/
│       ├── stock.py        # check_stock
│       ├── rfq.py          # send_rfqs
│       ├── quotes.py       # wait_for_quotes + extract_quotes
│       ├── history.py      # query_history
│       ├── evaluation.py   # evaluate_suppliers
│       ├── report.py       # generate_report
│       └── automation.py   # PO + PDF (post-approval only)
├── mcp_server.py           # FastMCP instance, registers all tools
├── services/
│   ├── gmail.py            # Gmail API wrapper
│   └── scoring.py          # deterministic scoring engine
├── api/
│   └── routes.py           # /chat SSE + /approve endpoints
├── core/
│   ├── state.py            # ProcurementState TypedDict
│   └── schemas.py
└── database/
    └── client.py
```

---

## Section 2 — FastMCP Tool Layer

FastMCP server defined in `mcp_server.py`. Each tool is a `@mcp.tool()` decorated function in `agents/tools/*.py`.

**Tool interfaces:**

| Tool | Reads from state | Writes to state |
|------|-----------------|-----------------|
| `check_stock` | `item_name`, `requested_qty` | `stock_sufficient`, `current_stock`, `item_id` |
| `send_rfqs` | `item_name`, `requested_qty`, `session_start_ts` | `rfq_sent_at`, `supplier_emails` |
| `wait_for_quotes` | `supplier_emails`, `rfq_sent_at` | `all_replied` |
| `extract_quotes` | `supplier_emails`, `rfq_sent_at` | `extracted_quotes[]` |
| `query_history` | `item_id` | `avg_unit_price_sen`, `avg_delivery_days` |
| `evaluate_suppliers` | `extracted_quotes[]`, `avg_unit_price_sen`, `avg_delivery_days` | `evaluated_suppliers[]` |
| `generate_report` | `evaluated_suppliers[]`, `stock_sufficient`, `item_name` | `report_markdown` |

**Key design decisions:**
- Planning call gives Gemini tool names + one-line descriptions only (not full schemas)
- Execution calls tools directly as Python functions (not via Gemini function calling)
- Each tool raises on failure → Manager catches, routes to error_node
- Automation Agent is NOT registered in FastMCP — triggered only by approve endpoint

---

## Section 3 — LangGraph Manager Agent

**Graph:**

```
START → plan_node → validate_node → stream_plan_node → execute_node → END
                         │                                   │
                    [invalid, retry]                   [step failure]
                         │                                   │
                    plan_node ──────────────────────── error_node → END
```

**Nodes:**

- `plan_node` — Gemini structured output. Prompt: user message + tool registry descriptions + ordering rules. Output: `{"plan": ["step1", ...]}`. On retry: includes validation error.
- `validate_node` — pure Python. Checks: (1) all steps in registry, (2) `extract_quotes` before `evaluate_suppliers`, (3) `evaluate_suppliers` before `generate_report`. Pass → continue. Fail + attempt 1 → back to plan_node. Fail + attempt 2 → error_node.
- `stream_plan_node` — pushes `plan` SSE event to session queue.
- `execute_node` — iterates `state["plan"]`, calls `TOOL_REGISTRY[step](state)` for each, streams `step_start`/`step_done` events, persists state to Supabase after each step.
- `error_node` — streams `error` event, persists `FAILED` status to Supabase.

**ProcurementState:**

```python
class ProcurementState(TypedDict, total=False):
    # Session
    session_id: str
    user_message: str
    session_start_ts: str
    plan: list[str]
    plan_attempts: int
    current_step: str
    status: str          # PLANNING | EXECUTING | AWAITING_APPROVAL | APPROVED | FAILED
    error: str | None

    # Populated by tools as plan executes
    item_name: str
    item_id: str
    requested_qty: int
    stock_sufficient: bool
    current_stock: int
    rfq_sent_at: str
    supplier_emails: list[str]
    all_replied: bool
    extracted_quotes: list[dict]
    avg_unit_price_sen: int
    avg_delivery_days: float
    evaluated_suppliers: list[dict]
    report_markdown: str
```

---

## Section 4 — SSE Chat Interface

**Endpoints:**

```
POST /api/v1/chat
     Body: { "message": "Buy 30 Dell XPS 15 laptops", "user_id": "..." }
     Returns: { "session_id": "sess_abc123" }

GET  /api/v1/chat/{session_id}/stream
     Returns: text/event-stream

POST /api/v1/chat/{session_id}/approve
     Returns: { "status": "SUCCESS", "po_pdf_url": "..." }
```

**SSE event types (in order):**

```
event: plan
data: {"steps": ["check_stock", "send_rfqs", ...]}

event: step_start
data: {"step": "check_stock", "message": "Checking stock levels..."}

event: step_done
data: {"step": "check_stock", "message": "Stock: 4 units (30 requested)"}

event: report
data: {"markdown": "## Procurement Evaluation Summary\n..."}

event: approve_ready
data: {"session_id": "sess_abc123", "message": "Approve to generate purchase order"}

event: error
data: {"step": "extract_quotes", "message": "No replies within 5 minutes"}
```

**SSE queue pattern:** `asyncio.Queue` per session. LangGraph pushes events; SSE endpoint drains. Sentinel `None` signals stream end.

Old endpoints (`POST /evaluations`, `GET /evaluations/{id}/status`, etc.) replaced entirely.

---

## Section 5 — Data Layer

**New `evaluations` table:**

```sql
CREATE TABLE evaluations (
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

**Updated `purchase_orders`:**

```sql
ALTER TABLE purchase_orders
    ADD COLUMN item_id    TEXT REFERENCES items(item_id),
    ADD COLUMN quantity   INT,
    ADD COLUMN pdf_url    TEXT;
```

**Persistence pattern:**
- Session start → `INSERT evaluations`
- After each step → `UPDATE evaluations SET current_step, state_json, updated_at`
- Report complete → `UPDATE evaluations SET status='AWAITING_APPROVAL', report_markdown`
- Approved → `UPDATE evaluations SET status='APPROVED'`

**New SupabaseRepository methods:**
```python
def create_evaluation(session_id: str, user_id: str) -> dict
def update_evaluation(session_id: str, **fields) -> dict
```

All existing methods unchanged.

---

## Tech Stack (final)

| Layer | Choice |
|-------|--------|
| Backend | FastAPI |
| Agent Graph | LangGraph |
| Tool Protocol | FastMCP (in-process) |
| LLM | Gemini (via `langchain-google-genai`) |
| PDF Parsing | `pypdf` |
| Email | Gmail API (Google API Python Client) |
| Database | Supabase (PostgreSQL) |
| File Storage | Supabase Storage |
| Streaming | SSE (`sse-starlette`) |
| Deployment | Docker + Docker Compose |

New dependencies to add to `pyproject.toml`:
- `fastmcp`
- `google-api-python-client`
- `google-auth-httplib2`
- `google-auth-oauthlib`
- `sse-starlette`
- `reportlab` (PDF generation)

---

## Out of Scope

- OCR (digital PDFs only)
- Frontend UI (SSE API only)
- OpenRouter, CrewAI, SQLAlchemy
- Reliability scoring, budget enforcement, inventory update on approval
- Multi-item evaluations (one item per session)
- Step retries on failure (abort and inform)
- MCP over HTTP transport (in-process only for MVP)
