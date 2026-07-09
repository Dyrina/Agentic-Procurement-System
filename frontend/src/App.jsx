import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Bot,
  CheckCircle2,
  CircleDot,
  Clock,
  FileText,
  Send,
  ShieldAlert,
  ShieldCheck,
  ShoppingCart,
  User,
} from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const USE_MOCK_STREAM = import.meta.env.VITE_USE_MOCK_STREAM === "true";

const EVENT_TYPES = [
  "plan",
  "step_start",
  "step_done",
  "report",
  "approve_ready",
  "error",
];

// Fixture matches the exact data shapes from the handoff doc's SSE table,
// in the documented sequence: plan -> (step_start/step_done) x N -> report -> approve_ready
const MOCK_EVENTS = [
  {
    type: "plan",
    data: {
      steps: [
        "check_stock",
        "send_rfqs",
        "wait_for_quotes",
        "extract_quotes",
        "query_history",
        "evaluate_suppliers",
        "generate_report",
      ],
      message:
        "My plan: check stock → send RFQs → wait for quotes → extract quotes → query history → evaluate suppliers → generate report.",
    },
  },
  { type: "step_start", data: { step: "check_stock", message: "Checking inventory stock levels..." } },
  { type: "step_done", data: { step: "check_stock", message: "✓ Completed" } },
  { type: "step_start", data: { step: "send_rfqs", message: "Sending RFQ emails to registered suppliers..." } },
  { type: "step_done", data: { step: "send_rfqs", message: "✓ Completed" } },
  { type: "step_start", data: { step: "wait_for_quotes", message: "Waiting for supplier quotation replies..." } },
  { type: "step_done", data: { step: "wait_for_quotes", message: "✓ Completed" } },
  { type: "step_start", data: { step: "extract_quotes", message: "Extracting quote details from supplier PDF attachments..." } },
  { type: "step_done", data: { step: "extract_quotes", message: "✓ Completed" } },
  { type: "step_start", data: { step: "query_history", message: "Querying historical purchase prices and delivery data..." } },
  { type: "step_done", data: { step: "query_history", message: "✓ Completed" } },
  { type: "step_start", data: { step: "evaluate_suppliers", message: "Evaluating suppliers using deterministic weighted scoring..." } },
  { type: "step_done", data: { step: "evaluate_suppliers", message: "✓ Completed" } },
  { type: "step_start", data: { step: "generate_report", message: "Generating recommendation report..." } },
  { type: "step_done", data: { step: "generate_report", message: "✓ Completed" } },
  {
    type: "report",
    data: {
      markdown: `# Procurement Recommendation Report

## Recommended Supplier
**Global IT Supplies** is recommended for this purchase.

## Reason
Although TechCorp Malaysia quoted a lower unit price, Global IT Supplies provides:
- Faster delivery time
- Better payment terms
- Lower operational risk
- Better overall procurement value

## Supplier Comparison

| Supplier | Unit Price | Delivery | Payment Terms | Score |
|---|---:|---:|---|---:|
| Global IT Supplies | RM 3,950.00 | 2 days | Net-60 | 92.5 |
| TechCorp Malaysia | RM 3,700.00 | 14 days | Net-30 | 64.0 |

## Final Decision
Proceed with **Global IT Supplies** subject to manager approval.`,
    },
  },
  {
    type: "approve_ready",
    data: {
      session_id: "sess_mock_001",
      message: "Recommendation report is ready. Please approve to generate PO.",
    },
  },
];

const PO_STORAGE_KEY = "procureai:purchase_orders";

function loadStoredPOs() {
  try {
    const raw = localStorage.getItem(PO_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveStoredPOs(pos) {
  try {
    localStorage.setItem(PO_STORAGE_KEY, JSON.stringify(pos));
  } catch {
    // storage unavailable — PO history just won't persist across reloads
  }
}

function getStepStatus(events, step) {
  const hasDone = events.some((e) => e.type === "step_done" && e.data.step === step);
  const hasStart = events.some((e) => e.type === "step_start" && e.data.step === step);
  if (hasDone) return "done";
  if (hasStart) return "active";
  return "pending";
}

function useProcurementStream(sessionId) {
  const [events, setEvents] = useState([]);
  const [report, setReport] = useState(null);
  const [canApprove, setCanApprove] = useState(false);
  const [streamError, setStreamError] = useState(null);

  useEffect(() => {
    if (!sessionId) return;

    setEvents([]);
    setReport(null);
    setCanApprove(false);
    setStreamError(null);

    if (USE_MOCK_STREAM) {
      let index = 0;
      const timer = setInterval(() => {
        const event = MOCK_EVENTS[index];
        if (!event) {
          clearInterval(timer);
          return;
        }
        setEvents((prev) => [...prev, event]);
        if (event.type === "report") setReport(event.data.markdown);
        if (event.type === "approve_ready") {
          setCanApprove(true);
          clearInterval(timer);
        }
        if (event.type === "error") {
          setStreamError(event.data.message);
          clearInterval(timer);
        }
        index += 1;
      }, 650);

      return () => clearInterval(timer);
    }

    // Real backend: routes are not live yet per the handoff doc (Task 17/18
    // pending). This branch is wired up so it's ready the moment they ship —
    // flip VITE_USE_MOCK_STREAM=false once /api/v1/chat/* exists.
    const streamUrl = `${BACKEND_URL}/api/v1/chat/${sessionId}/stream`;
    const es = new EventSource(streamUrl);

    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (event) => {
        const data = JSON.parse(event.data);
        setEvents((prev) => [...prev, { type, data }]);
        if (type === "report") setReport(data.markdown);
        if (type === "approve_ready") {
          setCanApprove(true);
          es.close();
        }
        if (type === "error") {
          setStreamError(data.message);
          es.close();
        }
      });
    }

    es.onerror = () => es.close();
    return () => es.close();
  }, [sessionId]);

  return { events, report, canApprove, streamError };
}

export default function App() {
  const [userId, setUserId] = useState("EMP-402");
  const [itemName, setItemName] = useState("Dell XPS 15 laptop");
  const [requestedQty, setRequestedQty] = useState(30);
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    {
      sender: "manager",
      text: "Manifest open. Describe what you need procured — I'll plan, execute, report, then wait for your approval.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [approveResult, setApproveResult] = useState(null);
  const [activeView, setActiveView] = useState("chat");
  const [purchaseOrders, setPurchaseOrders] = useState(() => loadStoredPOs());
  const chatWindowRef = useRef(null);

  const { events, report, canApprove, streamError } = useProcurementStream(sessionId);

  const planEvent = events.find((e) => e.type === "plan");
  const planSteps = planEvent?.data?.steps || [];

  useEffect(() => {
    const latest = events[events.length - 1];
    if (!latest) return;

<<<<<<< Updated upstream
    if (latest.type === "plan") addChat("manager", latest.data.message);
    if (latest.type === "step_start") addChat("manager", latest.data.message);
    if (latest.type === "step_done") addChat("manager", `${latest.data.step}: ${latest.data.message}`);
    if (latest.type === "report") addChat("manager", "Report generated. Review the recommendation below.");
=======
    if (latest.type === "report") {
      addChat("manager", "Report generated. Review the recommendation below.");
    }
>>>>>>> Stashed changes
    if (latest.type === "approve_ready") addChat("manager", latest.data.message);
    if (latest.type === "error") addChat("manager", `Error: ${latest.data.message}`);
  }, [events]);

  useEffect(() => {
    chatWindowRef.current?.scrollTo({ top: chatWindowRef.current.scrollHeight, behavior: "smooth" });
  }, [chatMessages, report, approveResult]);

  function addChat(sender, text) {
    setChatMessages((prev) => [...prev, { sender, text }]);
  }

  async function startSession(event) {
    event.preventDefault();
    if (!message.trim() || !itemName.trim() || !requestedQty) return;

    setLoading(true);
    setApproveResult(null);
    setSessionId(null);

    const userMessage = message.trim();
    addChat("user", userMessage);
    addChat("manager", "Opening procurement session...");

    // NOTE on the open question in the handoff doc: /api/v1/chat currently only
    // accepts free-text `message`, and nothing server-side extracts item/qty
    // from it yet, so check_stock will crash on a real message. Structured
    // item_name/requested_qty fields are captured here and sent alongside the
    // free text so the UI degrades gracefully either way this gets resolved —
    // but this still needs sign-off from the backend team before the contract
    // is final, per the doc.
    const payload = {
      message: userMessage,
      user_id: userId,
      item_name: itemName.trim(),
      requested_qty: Number(requestedQty),
    };

    try {
      if (USE_MOCK_STREAM) {
        setTimeout(() => {
          setSessionId("sess_mock_001");
          addChat("manager", "Mock session started.");
        }, 400);
      } else {
        const response = await fetch(`${BACKEND_URL}/api/v1/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) throw new Error(await response.text());

        const data = await response.json();
        setSessionId(data.session_id);
        addChat("manager", `Session started: ${data.session_id}`);
      }
    } catch (error) {
      addChat("manager", `Failed to start session: ${error.message}`);
    }

    setLoading(false);
  }

  async function approvePO() {
    if (!sessionId) return;

    setLoading(true);
    addChat("user", "Approve");
    addChat("manager", "Approval received. Generating Purchase Order...");

    try {
      let result;
      if (USE_MOCK_STREAM) {
        result = { status: "SUCCESS", po_pdf_url: "#" };
      } else {
        const response = await fetch(`${BACKEND_URL}/api/v1/chat/${sessionId}/approve`, {
          method: "POST",
        });
        if (!response.ok) throw new Error(await response.text());
        result = await response.json();
      }

      setApproveResult(result);
      addChat("manager", "Purchase Order generated successfully.");

      setPurchaseOrders((prev) => {
        const next = [
          {
            id: `${sessionId}-${Date.now()}`,
            sessionId,
            userId,
            itemName,
            requestedQty: Number(requestedQty),
            status: result.status,
            poPdfUrl: result.po_pdf_url,
            approvedAt: new Date().toISOString(),
          },
          ...prev,
        ];
        saveStoredPOs(next);
        return next;
      });
    } catch (error) {
      addChat("manager", `Approval failed: ${error.message}`);
    }

    setLoading(false);
  }

  const status = streamError
    ? "ERROR"
    : canApprove
      ? "AWAITING APPROVAL"
      : sessionId
        ? "RUNNING"
        : "IDLE";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <ShoppingCart size={22} strokeWidth={2} />
          <div>
            <h1>ProcureAI</h1>
            <span>Manifest Console</span>
          </div>
        </div>

        <nav>
          <button
            className={activeView === "chat" ? "active" : ""}
            onClick={() => setActiveView("chat")}
          >
            <Bot size={16} /> Manager Chat
          </button>
          <button
            className={activeView === "reports" ? "active" : ""}
            onClick={() => setActiveView("reports")}
          >
            <FileText size={16} /> Purchase Orders
            {purchaseOrders.length > 0 && <span className="nav-count">{purchaseOrders.length}</span>}
          </button>
          <button disabled>
            <ShieldCheck size={16} /> Audit Logs
          </button>
        </nav>

        <div className="mode-box">
          <span className="mode-label">Data source</span>
          <div className={`mode-pill ${USE_MOCK_STREAM ? "mock" : "live"}`}>
            <CircleDot size={12} />
            {USE_MOCK_STREAM ? "Mock SSE fixture" : "Live backend"}
          </div>
          {!USE_MOCK_STREAM && (
            <p className="mode-note">
              <ShieldAlert size={13} /> Routes 17/18 not shipped yet — calls will fail until they are.
            </p>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h2>Procurement Orchestrator</h2>
            <p>One message opens one session — plan, execute, report, approve.</p>
          </div>
          <div className="user-box">
            <label htmlFor="userId">Requester ID</label>
            <input id="userId" value={userId} onChange={(e) => setUserId(e.target.value)} />
          </div>
        </header>

        {activeView === "reports" ? (
          <section className="reports-view">
            <div className="panel-header">
              <h4>Purchase orders</h4>
              <span className="muted-text">{purchaseOrders.length} approved</span>
            </div>

            {purchaseOrders.length === 0 ? (
              <div className="reports-empty">
                <FileText size={28} />
                <p>No purchase orders yet.</p>
                <span className="muted-text">
                  Approve a report in Manager Chat and it'll show up here.
                </span>
              </div>
            ) : (
              <div className="po-list">
                {purchaseOrders.map((po) => (
                  <div className="po-card" key={po.id}>
                    <div className="po-card-main">
                      <h5>{po.itemName}</h5>
                      <p className="muted-text">
                        Qty {po.requestedQty} · requested by {po.userId}
                      </p>
                      <p className="po-meta">
                        <Clock size={13} />
                        {new Date(po.approvedAt).toLocaleString()}
                      </p>
                    </div>
                    <div className="po-card-side">
                      <span className={`status-chip status-${po.status.toLowerCase()}`}>{po.status}</span>
                      <code>{po.sessionId}</code>
                      <a href={po.poPdfUrl} target="_blank" rel="noreferrer">
                        Download PDF →
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        ) : (
        <div className="layout">
          <section className="chat-panel">
            <div className="panel-header">
              <h4>Procurement Manager Agent</h4>
              <span className={`status-chip status-${status.replace(/\s+/g, "-").toLowerCase()}`}>{status}</span>
            </div>

<<<<<<< Updated upstream
            <div className="chat-window" ref={chatWindowRef}>
              {chatMessages.map((chat, index) => (
                <div key={index} className={chat.sender === "user" ? "chat-row user" : "chat-row manager"}>
                  <div className="avatar">{chat.sender === "user" ? <User size={16} /> : <Bot size={16} />}</div>
                  <div className="bubble">{chat.text}</div>
=======
              <div className="chat-window" ref={chatWindowRef}>
                {chatMessages.map((chat, index) => (
                  <div key={index} className={chat.sender === "user" ? "chat-row user" : "chat-row manager"}>
                    <div className="avatar">{chat.sender === "user" ? <User size={16} /> : <Bot size={16} />}</div>
                    <div className="bubble">{chat.text}</div>
                  </div>
                ))}

                {awaitingInput && (
                  <div className="chat-row manager">
                    <div className="avatar"><Bot size={16} /></div>
                    <ReplyPrompt payload={awaitingInput} onReply={submitReply} disabled={replying} />
                  </div>
                )}

                {report && (
                  <div className="chat-row manager">
                    <div className="avatar"><Bot size={16} /></div>
                    <div className="bubble report-bubble">
                      <ReactMarkdown>{report}</ReactMarkdown>
                      {canApprove && !approveResult && (
                        <button className="approve-btn" onClick={approvePO} disabled={loading}>
                          Approve &amp; generate PO
                        </button>
                      )}
                      {completedMessage && (
                        <p className="completed-note">
                          <CheckCircle2 size={14} /> {completedMessage}
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {approveResult && (
                  <div className="chat-row manager">
                    <div className="avatar"><Bot size={16} /></div>
                    <div className="bubble success-bubble">
                      <h5>Purchase order generated</h5>
                      <p><b>Status:</b> {approveResult.status}</p>
                      <a href={approveResult.po_pdf_url} target="_blank" rel="noreferrer">Download PO PDF →</a>
                    </div>
                  </div>
                )}
              </div>

              <form className="chat-input" onSubmit={startSession}>
                <div className="message-row">
                  <input
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Describe the request, e.g. Buy 30 Dell XPS 15 laptops"
                    disabled={loading}
                  />
                  <button disabled={loading} type="submit" aria-label="Send">
                    <Send size={16} />
                  </button>
>>>>>>> Stashed changes
                </div>
              ))}

              {report && (
                <div className="chat-row manager">
                  <div className="avatar"><Bot size={16} /></div>
                  <div className="bubble report-bubble">
                    <ReactMarkdown>{report}</ReactMarkdown>
                    {canApprove && !approveResult && (
                      <button className="approve-btn" onClick={approvePO} disabled={loading}>
                        Approve &amp; generate PO
                      </button>
                    )}
                  </div>
                </div>
              )}

              {approveResult && (
                <div className="chat-row manager">
                  <div className="avatar"><Bot size={16} /></div>
                  <div className="bubble success-bubble">
                    <h5>Purchase order generated</h5>
                    <p><b>Status:</b> {approveResult.status}</p>
                    <a href={approveResult.po_pdf_url} target="_blank" rel="noreferrer">Download PO PDF →</a>
                  </div>
                </div>
              )}
            </div>

            <form className="chat-input" onSubmit={startSession}>
              <div className="structured-fields">
                <div className="field">
                  <label>Item</label>
                  <input value={itemName} onChange={(e) => setItemName(e.target.value)} placeholder="e.g. Dell XPS 15 laptop" />
                </div>
                <div className="field field-qty">
                  <label>Qty</label>
                  <input
                    type="number"
                    min="1"
                    value={requestedQty}
                    onChange={(e) => setRequestedQty(e.target.value)}
                  />
                </div>
              </div>
              <div className="message-row">
                <input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Describe the request, e.g. Buy 30 Dell XPS 15 laptops"
                />
                <button disabled={loading} type="submit" aria-label="Send">
                  <Send size={16} />
                </button>
              </div>
              <p className="field-note">
                Sent as structured fields alongside free text — pending backend confirmation on the
                item/qty extraction question flagged in the handoff doc.
              </p>
            </form>
          </section>

          <aside className="state-panel">
            <div className="card-box">
              <h5>Session</h5>
              <p><b>ID</b><code>{sessionId || "—"}</code></p>
              <p><b>Status</b><span className={`status-chip status-${status.replace(/\s+/g, "-").toLowerCase()}`}>{status}</span></p>
            </div>

            <div className="card-box">
              <h5>Execution manifest</h5>
              {planSteps.length === 0 && <p className="muted-text">No plan generated yet.</p>}
              {planSteps.map((step, index) => {
                const stepStatus = getStepStatus(events, step);
                return (
                  <div className={`plan-step ${stepStatus}`} key={step}>
                    <small>{String(index + 1).padStart(2, "0")}</small>
                    <span>{step}</span>
                    <CheckCircle2 size={15} />
                  </div>
                );
              })}
            </div>

            <div className="card-box log-box">
              <h5>Live event log</h5>
              {events.length === 0 && <p className="muted-text">Waiting for stream events...</p>}
              {events.map((event, index) => (
                <div className="event-row" key={index}>
                  <b>{event.type}</b>
                  <span>{event.data.message || event.data.step || event.data.session_id || "report received"}</span>
                </div>
              ))}
            </div>

            {streamError && (
              <div className="error-box">
                <ShieldAlert size={16} />
                <div>
                  <b>Stream error</b>
                  <p>{streamError}</p>
                </div>
              </div>
            )}
          </aside>
        </div>
        )}
      </main>
    </div>
  );
}