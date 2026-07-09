import { useState } from "react";

export default function ReplyPrompt({ payload, onReply, disabled }) {
  const [text, setText] = useState("");

  const questionText = payload.question || payload.message;

  function handleCandidateClick(candidate) {
    const label =
      candidate.item_id === "UNCATALOGED"
        ? candidate.name
        : `Selected: ${candidate.name} (${candidate.current_stock} in stock)`;
    onReply({ selected_item_id: candidate.item_id }, label);
  }

  function handleOptionClick(option) {
    onReply({ action: option }, `Action: ${option}`);
  }

  function handleTextSubmit(event) {
    event.preventDefault();
    if (!text.trim()) return;
    onReply({ text: text.trim() }, text.trim());
    setText("");
  }

  return (
    <div className="bubble reply-prompt">
      <p>{questionText}</p>

      {payload.candidates && (
        <div className="reply-options">
          {payload.candidates.map((candidate) => (
            <button
              key={candidate.item_id}
              className={
                candidate.item_id === "UNCATALOGED" ? "reply-option-btn reply-option-btn-muted" : "reply-option-btn"
              }
              disabled={disabled}
              onClick={() => handleCandidateClick(candidate)}
            >
              {candidate.item_id === "UNCATALOGED"
                ? candidate.name
                : `${candidate.name} · ${candidate.current_stock} in stock`}
            </button>
          ))}
        </div>
      )}

      {payload.options && (
        <div className="reply-options">
          {payload.options.map((option) => (
            <button
              key={option}
              className="reply-option-btn"
              disabled={disabled}
              onClick={() => handleOptionClick(option)}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      {!payload.candidates && !payload.options && (
        <form className="reply-text-form" onSubmit={handleTextSubmit}>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your answer..."
            disabled={disabled}
          />
          <button type="submit" disabled={disabled}>
            Send
          </button>
        </form>
      )}

      <button
        className="reply-option-btn reply-option-btn-muted"
        disabled={disabled}
        onClick={() => onReply({ action: "cancel" }, "Cancel request")}
      >
        Cancel request
      </button>
    </div>
  );
}
