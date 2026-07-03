# ProcureAI

## 1. Project Overview

### Problem statement

Procurement officers spend significant time on repetitive coordination work: sending Request for Quotation (RFQ) emails to suppliers, waiting for and collecting quotes, comparing them against historical pricing, evaluating suppliers across multiple criteria, preparing recommendation reports, and creating purchase orders. This work is manual, slow, and inconsistent between officers.

### Target users

- **Managers / requesters** — send a single chat message describing what they need (e.g. "Buy 30 Dell XPS 15 laptops") and receive a recommendation report to review.
- **Approving managers** — review the AI-generated supplier recommendation and click **Approve** before any purchase order is committed. Approval is a required human checkpoint, not a formality the system can bypass.
- **Backend/ops engineers** — operate and monitor the agent pipeline, Gmail integration, and Supabase data.

### System goal

Automate the end-to-end procurement evaluation workflow — from a single free-text request to a supplier recommendation — while keeping a human firmly in control of the final commitment. The system should never generate a purchase order or contact a supplier's commitment path without an explicit user approval action.

---

## 2. System Architecture

### Data flow (input → processing → output)

```
User message ("Buy 30 Dell XPS 15 laptops")
        │
        ▼
POST /api/v1/chat  →  session_id
        │
        ▼
Procurement Manager Agent (Gemini)
  1. Generate a structured JSON plan from the tool registry
  2. Validate the plan (registry check + ordering rules; one re-prompt if invalid)
  3. Stream the plan to the user over SSE
        │
        ▼
Execute plan steps in order, streaming step_start / step_done per step:
  check_stock → send_rfqs → wait_for_quotes → extract_quotes →
  query_history → evaluate_suppliers → generate_report
        │
        ▼
Stream `report` (markdown recommendation) + `approve_ready`
        │
        ▼
User clicks Approve  →  POST /api/v1/chat/{id}/approve
        │
        ▼
Automation Agent (outside the LLM plan, hardcoded, triggered only by approval)
  - Write PO record to Supabase
  - Generate PO PDF
  - Upload to Supabase Storage
  - Return download link → rendered in chat
```

Any step failure aborts the run, streams an `error` event, and preserves state in Supabase — there is no automatic retry.

### Module breakdown

| Module                            | Layer                   | Responsibility                                                                                                                    |
| --------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend (React + Vite)**       | Presentation            | Chat UI, SSE consumption, plan/step visualization, report rendering, approve action, purchase-order history view                  |
| **Procurement Manager Agent**     | Backend / orchestrator  | Parses the request, generates and validates the plan, drives execution, streams progress                                          |
| **Data Analyst Agent**            | Backend / worker        | `check_stock`, `query_history` — reads `items` and `purchase_history` tables                                                      |
| **Manager + Gmail integration**   | Backend / worker        | `send_rfqs` — drafts and sends RFQ emails via the Gmail API                                                                       |
| **Background poller**             | Backend / worker        | `wait_for_quotes` — polls Gmail every 15s for supplier replies, 5-minute timeout                                                  |
| **Document Agent**                | Backend / worker        | `extract_quotes` — pulls PDF attachments from reply emails, extracts text with `pypdf`, parses into structured quotes with Gemini |
| **Evaluation Agent**              | Backend / worker        | `evaluate_suppliers` — deterministic weighted scoring (price 55%, delivery 30%, payment terms 15%) plus rule-based risk flags     |
| **Reporting Agent**               | Backend / worker        | `generate_report` — assembles the markdown report from pipeline state, no LLM call                                                |
| **Automation Agent**              | Backend / post-approval | PO record creation, PDF generation, Storage upload — never part of an LLM-generated plan                                          |
| **Supabase (Postgres + Storage)** | Data                    | `suppliers`, `items`, `purchase_history`, `purchase_orders`, `evaluations`, `audit_logs`                                          |

---

## 3. Setup & Installation

### Prerequisites

- Docker and Docker Compose (recommended path), **or**
- Python 3.14 + [`uv`](https://docs.astral.sh/uv/) for the backend, and Node.js 18+ for the frontend, run locally
- A Supabase project (Postgres + Storage)
- A Google Cloud project with the Gemini API enabled and OAuth credentials for Gmail API access

### Environment setup

Two separate env files:

**Backend/root `.env`** (consumed by `docker-compose.yml`):

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-key
GOOGLE_API_KEY=your-google-api-key
GMAIL_CREDENTIALS_PATH=backend/tokens/credentials.json
GMAIL_TOKEN_PATH=backend/tokens/token.json
GMAIL_SENDER_EMAIL=your-gmail@gmail.com
```

**Frontend `frontend/.env`:**

```bash
VITE_BACKEND_URL=http://localhost:8000
VITE_USE_MOCK_STREAM=true
```

`VITE_USE_MOCK_STREAM=true` runs the chat UI entirely off a local mock SSE fixture, with no backend required — this is the current default since the backend routes below are still being finalized.

### Dependencies

| Layer    | Manager                             | Key packages                                                                                                             |
| -------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Backend  | `uv` (`pyproject.toml` / `uv.lock`) | fastapi, uvicorn, langgraph, langchain-google-genai, supabase, pypdf, google-api-python-client, sse-starlette, reportlab |
| Frontend | npm (`package.json`)                | react, react-dom, react-markdown, lucide-react, vite                                                                     |

### How to run

**With Docker Compose (recommended):**

```bash
cp .env.example .env      # fill in Supabase / Gemini / Gmail values
docker compose up --build
```

- Backend: `http://localhost:8000` (health check at `/health`)
- Frontend: `http://localhost:5173` (Nginx-served build, container port 80 mapped)

**Locally, without Docker:**

```bash
# backend
cd backend
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# frontend, in a second terminal
cd frontend
npm install
npm run dev
```

**Database:** run `schema.sql` once in the Supabase SQL Editor — it creates all tables and inserts seed suppliers/items/purchase history so the agents have data to reason over immediately.

**Gmail:** enable the Gmail API in Google Cloud Console, download `credentials.json` into `backend/tokens/`, and complete the OAuth consent flow on first run to generate `token.json`.

---

## 4. Features

| Feature                                    | Status                                                         | Explanation                                                                                                                                                                                                                                        |
| ------------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dynamic plan generation**                | Designed, backend pending                                      | The Manager Agent produces a different ordered step sequence per request type (e.g. a stock-check-only question skips RFQs entirely), instead of running one fixed pipeline every time.                                                            |
| **Plan validation with bounded re-prompt** | Designed, backend pending                                      | Generated plans are checked against the tool registry and dependency rules (`extract_quotes` before `evaluate_suppliers` before `generate_report`); an invalid plan gets exactly one Gemini re-prompt before the request fails with a clear error. |
| **Streaming chat UI**                      | Implemented (frontend)                                         | React chat interface consumes an SSE stream (`plan`, `step_start`, `step_done`, `report`, `approve_ready`, `error`) and renders each event as a conversational message in real time.                                                               |
| **Execution manifest panel**               | Implemented (frontend)                                         | A live checklist derived from the `plan` event shows each step's status (pending / active / done) as `step_start`/`step_done` events arrive.                                                                                                       |
| **Raw live event log**                     | Implemented (frontend)                                         | A secondary panel shows every SSE event verbatim, for debugging what the stream actually sent.                                                                                                                                                     |
| **Markdown recommendation report**         | Implemented (frontend rendering) / Designed (backend assembly) | The Reporting Agent assembles an executive summary, supplier comparison table, and decision explanation from pipeline state (no LLM call); the frontend renders it with `react-markdown`.                                                          |
| **Human-in-the-loop approval**             | Implemented (frontend) / Designed (backend)                    | No purchase order or supplier commitment happens without an explicit **Approve** click; the Automation Agent is architecturally excluded from the LLM's plan and can only be reached via this action.                                              |
| **Deterministic supplier scoring**         | Designed, backend pending                                      | Weighted scoring (price 55%, delivery 30%, payment terms 15%) plus rule-based risk flags (e.g. price >10% above historical average) — no LLM judgment involved, for reproducibility.                                                               |
| **Purchase order history (Reports view)**  | Implemented (frontend, local only)                             | Every approved PO is recorded client-side (`localStorage`) and listed with item, quantity, requester, timestamp, and PDF link. This is a stopgap — see Limitations.                                                                                |
| **Mock/live data source toggle**           | Implemented (frontend)                                         | `VITE_USE_MOCK_STREAM` switches the entire chat flow between a local fixture and the real backend without code changes, so frontend work isn't blocked on backend delivery.                                                                        |
| **Gmail RFQ send/receive**                 | Designed, backend pending                                      | Outbound RFQ emails to registered supplier addresses; inbound polling for replies (15s interval, 5-minute timeout) and PDF quote extraction via `pypdf` + Gemini structured output.                                                                |
| **Audit trail**                            | Placeholder only                                               | A disabled nav item; the `audit_logs` table exists in the schema but no endpoint or UI reads from it yet.                                                                                                                                          |

---

## 5. Technical Decisions

### Architecture choices

- **Plan-Then-Execute over a free-running agent loop.** The Manager Agent produces a complete plan up front and validates it before executing, rather than deciding each next action step-by-step (ReAct-style). This makes the pipeline auditable and interruptible, and lets the frontend show the user the full plan before any work happens.
- **Automation Agent excluded from the tool registry.** Keeping PO generation entirely outside what the LLM can plan is a deliberate safety boundary — no prompt injection or planning error can cause a commitment action; only an explicit user click can.
- **Deterministic scoring instead of LLM-judged evaluation.** Supplier scoring uses a fixed weighted formula and lookup tables rather than asking Gemini to "pick the best supplier." This trades some flexibility for reproducibility and explainability — the same quotes always produce the same ranking, and the reasoning can be shown to the user directly.
- **SSE over WebSockets for streaming.** The chat stream only needs server→client updates, so Server-Sent Events (via native `EventSource` and `sse-starlette`) were chosen over WebSockets for simplicity — no custom protocol, automatic reconnection handling, works over plain HTTP.
- **Mock-first frontend development.** The frontend was built against the documented SSE contract with a local fixture (`VITE_USE_MOCK_STREAM`) rather than waiting on backend delivery, so both sides could progress in parallel from a shared spec.
- **Supabase for both database and file storage.** Using one provider for Postgres and PDF storage reduces infrastructure surface area for what is currently a single-service backend.

### Trade-offs made

- **One item per session, no multi-item requests.** Simplifies the plan, the state schema, and the PO structure considerably, at the cost of not supporting a single request like "30 laptops and 10 chairs" — that would need two separate sessions today.
- **No step retries.** A failed step aborts the whole run and reports the error, rather than attempting automatic recovery. Simpler and more predictable failure behavior, at the cost of resilience — a transient Gmail API hiccup currently kills the whole session.
- **Free-text request parsing.** Accepting a single free-text message is a better user experience than a structured form, but pushes item/quantity extraction onto the LLM/backend rather than a guaranteed-valid form input — see Limitations.
- **Bounded plan re-prompting (one retry).** Re-prompting Gemini once on an invalid plan balances self-correction against unbounded retries and cost; a persistently invalid plan surfaces as a user-facing error instead of looping.
- **Client-side PO history instead of a backend-backed one.** Storing approved POs in `localStorage` was the fastest way to make the Reports view usable before a real `GET /purchase-orders` endpoint exists, at the clear cost of not being the source of truth — see Limitations.

---

## 6. Limitations

### Known issues

- **The frontend/backend contract isn't finalized.** Free-text requests don't yet have a confirmed way to resolve a structured `item_id` before `check_stock` runs, the chat/stream/approve routes aren't confirmed live, and CORS/auth aren't configured — the frontend defaults to a mock stream until these are settled with the backend team.
- **Reports and Audit Trail aren't backend-backed yet.** Purchase order history is stored client-side in `localStorage` (not fetched from Supabase, so it won't show POs approved elsewhere), and Audit Trail has no endpoint or UI wired up at all despite the `audit_logs` table existing in the schema.

### Future improvements

- Add a real item lookup/autocomplete endpoint so the frontend can send a validated `item_id` instead of guessing from free text.
- Add `GET /api/v1/purchase-orders` and `GET /api/v1/audit-logs` endpoints, and point the Reports and Audit Trail views at them instead of local-only/placeholder state.
- Add authentication and finalize CORS configuration before any non-local deployment.
- Support multi-item requests within a single session.
- Add step-level retry/resume instead of hard-aborting the whole session on a transient failure.
- Add OCR fallback for scanned supplier quotes.
- Add budget enforcement and supplier reliability scoring to the evaluation weighting (both explicitly out of scope for the current MVP).
- Add automatic inventory updates on PO approval (currently out of scope — stock levels are read-only).
