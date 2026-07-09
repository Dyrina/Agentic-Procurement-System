import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  CheckCircle2,
  Clock,
  FileText,
  Send,
  ShieldAlert,
  ShieldCheck,
  ShoppingCart,
  User,
} from "lucide-react";
import ReplyPrompt from "./ReplyPrompt";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const EVENT_TYPES = ["progress", "awaiting_input", "report", "approve_ready", "completed", "error"];



function useProcurementStream(sessionId) {
  const [events, setEvents] = useState([]);
  const [report, setReport] = useState(null);
  const [canApprove, setCanApprove] = useState(false);
  const [completedMessage, setCompletedMessage] = useState(null);
  const [awaitingInput, setAwaitingInput] = useState(null);
  const [streamError, setStreamError] = useState(null);

  useEffect(() => {
    setEvents([]);
    setReport(null);
    setCanApprove(false);
    setCompletedMessage(null);
    setAwaitingInput(null);
    setStreamError(null);

    if (!sessionId) return;

    const streamUrl = `${BACKEND_URL}/api/v1/chat/${sessionId}/stream`;
    const es = new EventSource(streamUrl);

    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (event) => {
        const data = JSON.parse(event.data);
        setEvents((prev) => [...prev, { type, data }]);

        // The stream stays open across a pause — every non-awaiting_input event
        // means the pause (if any) is over, so clear it.
        setAwaitingInput(type === "awaiting_input" ? data : null);

        if (type === "report") setReport(data.markdown);
        if (type === "approve_ready") {
          setCanApprove(true);
          es.close();
        }
        if (type === "completed") {
          setCompletedMessage(data.message);
          es.close();
        }
        if (type === "error") {
          setStreamError(data.message);
          es.close();
        }
      });
    }

    // Don't close on error — EventSource auto-reconnects after network blips, and the
    // backend replays session state on reconnect. Terminal events above close it for real.
    return () => es.close();
  }, [sessionId]);

  return { events, report, canApprove, completedMessage, awaitingInput, setAwaitingInput, streamError, setStreamError };
}

export default function App() {
  const [userId, setUserId] = useState("Manager");
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [currentRequestText, setCurrentRequestText] = useState("");
  const [chatMessages, setChatMessages] = useState([
    {
      sender: "manager",
      text: "Manifest open. Describe what you need procured — I'll check stock, source quotes, evaluate suppliers, and wait for your approval.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [replying, setReplying] = useState(false);
  const [approveResult, setApproveResult] = useState(null);
  const [activeView, setActiveView] = useState("chat");
  const [processedEventsCount, setProcessedEventsCount] = useState(0);
  const [purchaseOrders, setPurchaseOrders] = useState([]);
  const chatWindowRef = useRef(null);
  const eventLogRef = useRef(null);

  const { events, report, canApprove, completedMessage, awaitingInput, setAwaitingInput, streamError, setStreamError } =
    useProcurementStream(sessionId);

  useEffect(() => {
    async function fetchPOs() {
      if (!userId) return;
      try {
        // If the user is a "Manager", fetch all POs globally. Otherwise, filter by their specific ID.
        const queryParams = userId.toLowerCase() === "manager" ? "" : `?user_id=${encodeURIComponent(userId)}`;
        const response = await fetch(`${BACKEND_URL}/api/v1/purchase-orders${queryParams}`);
        if (response.ok) {
          const data = await response.json();
          const mapped = data.map((dbPo) => ({
            id: dbPo.new_po_id,
            requestText: `${dbPo.item_name || dbPo.item_id} (Qty: ${dbPo.quantity || 1})`,
            userId: dbPo.approved_by || "System",
            status: dbPo.status || "APPROVED",
            poNumber: dbPo.new_po_id,
            poPdfUrl: dbPo.pdf_url,
            approvedAt: dbPo.created_at || new Date().toISOString(),
          }));
          setPurchaseOrders(mapped);
        }
      } catch (error) {
        console.error("Failed to load POs:", error);
      }
    }
    fetchPOs();
  }, [userId]);

  useEffect(() => {
    if (events.length === 0 && processedEventsCount > 0) {
      setProcessedEventsCount(0);
      return;
    }

    if (events.length > processedEventsCount) {
      const newEvents = events.slice(processedEventsCount);
      setProcessedEventsCount(events.length);

      newEvents.forEach((latest) => {
        if (latest.type === "awaiting_input") {
          setChatMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.payload && last.payload.type === latest.data.type) return prev;
            return [...prev, { sender: "manager", text: null, payload: latest.data }];
          });
        }
        if (latest.type === "report") {
          setChatMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.payload && last.payload.type === "report") return prev;
            return [...prev, { sender: "manager", text: null, payload: { type: "report", markdown: latest.data.markdown } }];
          });
        }
        if (latest.type === "approve_ready") {
          setChatMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.payload && last.payload.type === "approve_ready") return prev;
            return [...prev, { sender: "manager", text: null, payload: { type: "approve_ready", message: latest.data.message } }];
          });
        }
        if (latest.type === "completed") {
          setChatMessages((prev) => [...prev, { sender: "manager", text: latest.data.message, payload: null }]);
        }
        if (latest.type === "error") {
          setChatMessages((prev) => [...prev, { sender: "manager", text: `Error: ${latest.data.message}`, payload: null }]);
        }
      });
    }
  }, [events, processedEventsCount]);

  useEffect(() => {
    chatWindowRef.current?.scrollTo({ top: chatWindowRef.current.scrollHeight, behavior: "smooth" });
  }, [chatMessages, report, approveResult, awaitingInput]);

  useEffect(() => {
    eventLogRef.current?.scrollTo({ top: eventLogRef.current.scrollHeight, behavior: "smooth" });
  }, [events]);

  function addChat(sender, text, payload = null) {
    setChatMessages((prev) => [...prev, { sender, text, payload }]);
  }

  async function startSession(event) {
    event.preventDefault();
    if (!message.trim()) return;

    setLoading(true);
    setApproveResult(null);
    setSessionId(null);

    const userMessage = message.trim();
    setCurrentRequestText(userMessage);
    
    // Clear out the previous session's chat log and start fresh
    setChatMessages([
      { sender: "user", text: userMessage, payload: null },
      { sender: "manager", text: "Opening procurement session...", payload: null }
    ]);

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, user_id: userId }),
      });

      if (!response.ok) throw new Error(await response.text());

      const data = await response.json();
      setSessionId(data.session_id);
      addChat("manager", `Session started: ${data.session_id}`);
      setMessage("");
    } catch (error) {
      addChat("manager", `Failed to start session: ${error.message}`);
    }

    setLoading(false);
  }

  async function submitReply(replyPayload, label) {
    if (!sessionId) return;

    setReplying(true);
    addChat("user", label);
    setAwaitingInput(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/chat/${sessionId}/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reply: replyPayload }),
      });
      if (!response.ok) throw new Error(await response.text());
    } catch (error) {
      addChat("manager", `Reply failed: ${error.message}`);
    }

    setReplying(false);
  }

  async function approvePO() {
    if (!sessionId) return;

    setLoading(true);
    addChat("user", "Approve");
    addChat("manager", "Approval received. Generating Purchase Order...");

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/chat/${sessionId}/approve`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();

      setApproveResult(result);
      addChat("manager", null, { type: "approve_result", data: result });

      setPurchaseOrders((prev) => [
        {
          id: `${sessionId}-${Date.now()}`,
          requestText: currentRequestText,
          userId: userId,
          status: result.status || "APPROVED",
          poNumber: result.po_number,
          poPdfUrl: result.po_pdf_url,
          approvedAt: new Date().toISOString(),
        },
        ...prev,
      ]);
    } catch (error) {
      addChat("manager", `Approval failed: ${error.message}`);
      setStreamError(`Approval failed: ${error.message}`);
    }

    setLoading(false);
  }

  const status = streamError
    ? "ERROR"
    : completedMessage
      ? (completedMessage.toLowerCase().includes("cancel") ? "CANCELLED" : "COMPLETED")
      : approveResult
        ? "APPROVED"
        : canApprove
          ? "AWAITING APPROVAL"
          : awaitingInput
            ? "AWAITING INPUT"
            : sessionId
              ? "RUNNING"
              : "IDLE";

  const isSessionActive = ["RUNNING", "AWAITING INPUT", "AWAITING APPROVAL"].includes(status);

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
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h2>Procurement Orchestrator</h2>
            <p>One message opens one session — check stock, source, evaluate, report, approve.</p>
          </div>
          <div className="user-box">
            <label htmlFor="userId">Requester ID</label>
            <input 
              id="userId" 
              value={userId} 
              onChange={(e) => setUserId(e.target.value)} 
              disabled={isSessionActive} 
            />
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
                      <h5>{po.requestText || "Purchase order"}</h5>
                      <p className="muted-text">requested by {po.userId}</p>
                      <p className="po-meta">
                        <Clock size={13} />
                        {new Date(po.approvedAt).toLocaleString()}
                      </p>
                    </div>
                    <div className="po-card-side">
                      <span className={`status-chip status-${po.status.toLowerCase()}`}>{po.status}</span>
                      <code>{po.poNumber || po.sessionId}</code>
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

              <div className="chat-window" ref={chatWindowRef}>
                {chatMessages.map((chat, index) => (
                  <div key={index} className={chat.sender === "user" ? "chat-row user" : "chat-row manager"}>
                    <div className="avatar">{chat.sender === "user" ? <User size={16} /> : <Bot size={16} />}</div>
                    {chat.payload ? (
                      chat.payload.type === "report" ? (
                        <div className="bubble report-bubble">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{chat.payload.markdown}</ReactMarkdown>
                        </div>
                      ) : chat.payload.type === "approve_ready" ? (
                        <div className="bubble">
                          <p style={{ margin: "0 0 10px" }}>{chat.payload.message}</p>
                          <button className="approve-btn" onClick={approvePO} disabled={loading || !!approveResult || index !== chatMessages.length - 1}>
                            Approve &amp; generate PO
                          </button>
                        </div>
                      ) : chat.payload.type === "approve_result" ? (
                        <div className="bubble success-bubble">
                          <h5>Purchase order generated</h5>
                          <p><b>Status:</b> {chat.payload.data.status}</p>
                          <a href={chat.payload.data.po_pdf_url} target="_blank" rel="noreferrer">Download PO PDF &rarr;</a>
                        </div>
                      ) : (
                        <ReplyPrompt 
                          payload={chat.payload} 
                          onReply={submitReply} 
                          disabled={replying || index !== chatMessages.length - 1} 
                        />
                      )
                    ) : (
                      <div className="bubble">{chat.text}</div>
                    )}
                  </div>
                ))}
              </div>

              <form className="chat-input" onSubmit={startSession}>
                <div className="message-row">
                  <input
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Describe the request, e.g. Buy 30 Dell XPS 15 laptops"
                    disabled={loading || isSessionActive}
                  />
                  <button disabled={loading || isSessionActive} type="submit" aria-label="Send">
                    <Send size={16} />
                  </button>
                </div>
              </form>
            </section>

            <aside className="state-panel">
              <div className="card-box">
                <h5>Session</h5>
                <p><b>ID</b><code>{sessionId || "—"}</code></p>
                <p><b>Status</b><span className={`status-chip status-${status.replace(/\s+/g, "-").toLowerCase()}`}>{status}</span></p>
              </div>

              <div className="card-box log-box" ref={eventLogRef}>
                <h5>Live event log</h5>
                {events.length === 0 && <p className="muted-text">Waiting for stream events...</p>}
                {events.map((event, index) => (
                  <div className="event-row" key={index}>
                    <b>{event.type}</b>
                    <span>{event.data.message || event.data.question || event.data.session_id || "event received"}</span>
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