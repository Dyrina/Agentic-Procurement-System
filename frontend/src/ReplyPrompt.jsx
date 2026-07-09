import { useState } from "react";

export default function ReplyPrompt({ payload, onReply, disabled }) {
  const [text, setText] = useState("");
  const [clickedItem, setClickedItem] = useState(null);

  const questionText = payload.question || payload.message;

  function handleCandidateClick(candidate) {
    setClickedItem(candidate.item_id);
    const label =
      candidate.item_id === "UNCATALOGED"
        ? candidate.name
        : `Selected: ${candidate.name} (${candidate.current_stock} in stock)`;
    onReply({ selected_item_id: candidate.item_id }, label);
  }

  function handleOptionClick(option) {
    setClickedItem(option);
    onReply({ action: option }, `Action: ${option}`);
  }

  function handleTextSubmit(event) {
    event.preventDefault();
    if (!text.trim()) return;
    setClickedItem("text");
    onReply({ text: text.trim() }, text.trim());
    setText("");
  }

  function handleCancelClick() {
    setClickedItem("cancel");
    onReply({ action: "cancel" }, "Cancel request");
  }

  return (
    <div className="bubble reply-prompt">
      <p>{questionText}</p>

      {payload.candidates && (
        <div className="reply-options">
          {payload.candidates.map((candidate) => (
            <button
              key={candidate.item_id}
              className={`reply-option-btn ${candidate.item_id === "UNCATALOGED" ? "reply-option-btn-muted" : ""} ${clickedItem === candidate.item_id ? "selected" : ""}`}
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
              className={`reply-option-btn ${clickedItem === option ? "selected" : ""}`}
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
          <button type="submit" disabled={disabled} className={clickedItem === "text" ? "selected" : ""}>
            Send
          </button>
        </form>
      )}

      <button
        className={`reply-option-btn reply-option-btn-muted ${clickedItem === "cancel" ? "selected" : ""}`}
        disabled={disabled}
        onClick={handleCancelClick}
      >
        Cancel request
      </button>
    </div>
  );
}
