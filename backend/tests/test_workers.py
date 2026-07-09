"""Unit tests for the supervisor-loop worker agents (src/agents/workers/*.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.workers import _last_tool_call
from src.core.state import ProcurementState

# ── shared helpers ───────────────────────────────────────────────────────────


def test_last_tool_call_finds_matching_call():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_items", "args": {"query": "laptop"}, "id": "1"}],
        ),
    ]
    assert _last_tool_call(messages, "search_items") == {"query": "laptop"}


def test_last_tool_call_returns_most_recent_when_called_twice():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "submit_intake", "args": {"item_name": "old"}, "id": "1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "submit_intake", "args": {"item_name": "new"}, "id": "2"}],
        ),
    ]
    assert _last_tool_call(messages, "submit_intake") == {"item_name": "new"}


def test_last_tool_call_returns_none_when_absent():
    messages = [HumanMessage(content="hi"), AIMessage(content="no tools here")]
    assert _last_tool_call(messages, "submit_intake") is None


# ── intake ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_intake_node_submits_parsed_request(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {"item_name": "Dell XPS 15 Laptop", "requested_qty": 30},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "order 30 units of Dell XPS 15 laptop",
    }
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["item_name"] == "Dell XPS 15 Laptop"
    assert result["requested_qty"] == 30
    assert result["needs_clarification"] is False
    assert "error" not in result


@pytest.mark.asyncio
async def test_intake_node_flags_ambiguous_request(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "flag_ambiguous",
                        "args": {"question": "How many units do you need?"},
                        "id": "1",
                    }
                ],
            ),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "I need some laptops"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["needs_clarification"] is True
    assert result["clarification_payload"]["question"] == "How many units do you need?"
    assert result["intake_attempts"] == 1


@pytest.mark.asyncio
async def test_intake_node_fails_after_max_ambiguous_attempts(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "flag_ambiguous",
                        "args": {"question": "still unclear"},
                        "id": "1",
                    }
                ],
            ),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "laptops", "intake_attempts": 2}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["needs_clarification"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_intake_node_catches_exceptions():
    from src.agents.workers.intake import intake_node

    state: ProcurementState = {"session_id": "s1", "user_message": "laptops"}
    with patch("src.agents.workers.intake._build_llm", side_effect=RuntimeError("boom")):
        result = await intake_node(state)

    assert result["error"] == "boom"
    assert "FAILED" in result["supervisor_history"][-1]["summary"]


@pytest.mark.asyncio
async def test_intake_await_node_passthrough_when_not_needed():
    from src.agents.workers.intake import intake_await_node

    state: ProcurementState = {"session_id": "s1", "needs_clarification": False}
    result = await intake_await_node(state)
    assert result == state


@pytest.mark.asyncio
async def test_intake_await_node_resumes_with_clarified_message():
    from src.agents.workers.intake import intake_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "laptops",
        "needs_clarification": True,
        "clarification_payload": {"type": "intake_clarification", "question": "how many?"},
    }
    with patch("src.agents.workers.intake.interrupt", return_value="30 units"):
        result = await intake_await_node(state)

    assert result["needs_clarification"] is False
    assert result["clarification_payload"] is None
    assert "30 units" in result["user_message"]


# ── inventory ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inventory_node_always_asks_for_confirmation(fake_llm):
    from src.agents.workers.inventory import inventory_node

    candidates = [{"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "similarity": 0.9}]
    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_items",
                        "args": {"query": "Dell XPS 15"},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_user_to_confirm",
                        "args": {
                            "candidates": candidates,
                            "question": "Did you mean Dell XPS 15 Laptop?",
                        },
                        "id": "2",
                    }
                ],
            ),
        ]
    )
    mock_db = MagicMock()
    mock_db.rpc.return_value = candidates
    state: ProcurementState = {"session_id": "s1", "item_name": "Dell XPS 15", "requested_qty": 30}
    with (
        patch("src.agents.workers.inventory._build_llm", return_value=llm),
        patch("src.agents.workers.inventory.SupabaseRepository", return_value=mock_db),
    ):
        result = await inventory_node(state)

    from src.agents.workers.inventory import _NOT_LISTED_OPTION

    assert result["needs_clarification"] is True
    # The "none of these — new item" escape hatch is always appended after real candidates.
    assert result["clarification_payload"]["candidates"] == [*candidates, _NOT_LISTED_OPTION]
    assert "item_id" not in result  # never silently picked
    # Question wording is deterministic and intent-aware — never the LLM's ("purchase?" on a
    # stock check was a live bug). No intent in this state → default (purchase-path) wording.
    assert result["clarification_payload"]["question"] != "Did you mean Dell XPS 15 Laptop?"
    assert "check stock" not in result["clarification_payload"]["question"]


@pytest.mark.asyncio
async def test_inventory_confirm_question_matches_check_stock_intent(fake_llm):
    from src.agents.workers.inventory import inventory_node

    candidates = [{"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "similarity": 0.8}]
    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_user_to_confirm",
                        "args": {
                            "candidates": candidates,
                            "question": "Is this the one you would like to purchase?",
                        },
                        "id": "1",
                    }
                ],
            ),
        ]
    )
    mock_db = MagicMock()
    mock_db.rpc.return_value = candidates
    state: ProcurementState = {
        "session_id": "s1",
        "item_name": "Dell XPS 15",
        "requested_qty": 0,
        "intent": "check_stock",
    }
    with (
        patch("src.agents.workers.inventory._build_llm", return_value=llm),
        patch("src.agents.workers.inventory.SupabaseRepository", return_value=mock_db),
    ):
        result = await inventory_node(state)

    question = result["clarification_payload"]["question"]
    assert "check stock" in question
    assert "purchase" not in question  # the LLM's purchase wording must not leak through


@pytest.mark.asyncio
async def test_inventory_node_fails_if_agent_skips_confirmation(fake_llm):
    from src.agents.workers.inventory import inventory_node

    llm = fake_llm([AIMessage(content="Dell XPS 15 Laptop it is.")])  # no tool call at all
    state: ProcurementState = {"session_id": "s1", "item_name": "Dell XPS 15", "requested_qty": 30}
    with patch("src.agents.workers.inventory._build_llm", return_value=llm):
        result = await inventory_node(state)

    assert "error" in result


@pytest.mark.asyncio
async def test_inventory_node_checks_stock_once_item_id_confirmed():
    from src.agents.workers.inventory import inventory_node

    mock_db = MagicMock()
    mock_db.get_item.return_value = {"item_id": "IT-XPS-15", "current_stock": 4}
    state: ProcurementState = {"session_id": "s1", "item_id": "IT-XPS-15", "requested_qty": 30}
    with patch("src.agents.workers.inventory.SupabaseRepository", return_value=mock_db):
        result = await inventory_node(state)

    assert result["current_stock"] == 4
    assert result["stock_sufficient"] is False


@pytest.mark.asyncio
async def test_intake_node_rejects_out_of_scope_request(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reject_out_of_scope",
                        "args": {"reason": "I can only help with procurement requests."},
                        "id": "1",
                    }
                ],
            ),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "write me a poem"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["completion_message"] == "I can only help with procurement requests."
    assert result["needs_clarification"] is False
    assert "error" not in result


@pytest.mark.asyncio
async def test_intake_node_extracts_intent(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {"item_name": "Laptop", "requested_qty": 2, "intent": "check_stock"},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "how many laptops do we have?"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["intent"] == "check_stock"


@pytest.mark.asyncio
async def test_intake_node_unknown_intent_degrades_to_buy(fake_llm):
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {"item_name": "Laptop", "requested_qty": 2, "intent": "banana"},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "2 laptops"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["intent"] == "buy"


@pytest.mark.asyncio
async def test_intake_node_purchase_intent_with_zero_qty_asks_instead_of_proceeding(fake_llm):
    """Backstop: a buy with no real quantity must clarify — an invented qty becomes an RFQ/PO."""
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {"item_name": "Laptop", "requested_qty": 0, "intent": "buy"},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "buy laptops"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["needs_clarification"] is True
    assert "how many" in result["clarification_payload"]["question"].lower()
    assert "item_name" not in result


@pytest.mark.asyncio
async def test_intake_node_check_stock_with_zero_qty_proceeds(fake_llm):
    """qty 0 is the normal shape of 'how many X do we have?' — must pass straight through."""
    from src.agents.workers.intake import intake_node

    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_intake",
                        "args": {"item_name": "Laptop", "requested_qty": 0, "intent": "check_stock"},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "user_message": "how many laptops in stock?"}
    with patch("src.agents.workers.intake._build_llm", return_value=llm):
        result = await intake_node(state)

    assert result["requested_qty"] == 0
    assert result["intent"] == "check_stock"
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_intake_await_node_cancel_sets_cancelled():
    from src.agents.workers.intake import intake_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "user_message": "laptops",
        "needs_clarification": True,
        "clarification_payload": {"type": "intake_clarification", "question": "how many?"},
    }
    with patch("src.agents.workers.intake.interrupt", return_value={"action": "cancel"}):
        result = await intake_await_node(state)

    assert result["cancelled"] is True
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_inventory_await_node_cancel_sets_cancelled():
    from src.agents.workers.inventory import inventory_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "needs_clarification": True,
        "clarification_payload": {"type": "inventory_candidate_confirm", "candidates": [], "question": "?"},
    }
    with patch("src.agents.workers.inventory.interrupt", return_value={"action": "cancel"}):
        result = await inventory_await_node(state)

    assert result["cancelled"] is True
    assert "item_id" not in result


@pytest.mark.asyncio
async def test_sourcing_await_node_cancel_sets_cancelled():
    from src.agents.workers.sourcing import sourcing_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "needs_clarification": True,
        "clarification_payload": {"type": "sourcing_timeout", "options": []},
        "supplier_emails": ["a@b.com"],
        "rfq_sent_at": "2026-07-07T00:00:00Z",
    }
    with patch("src.agents.workers.sourcing.interrupt", return_value={"action": "cancel"}):
        result = await sourcing_await_node(state)

    assert result["cancelled"] is True


@pytest.mark.asyncio
async def test_inventory_await_node_resumes_with_selected_item_id():
    from src.agents.workers.inventory import inventory_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "needs_clarification": True,
        "clarification_payload": {
            "type": "inventory_candidate_confirm",
            "candidates": [],
            "question": "?",
        },
    }
    with patch(
        "src.agents.workers.inventory.interrupt", return_value={"selected_item_id": "IT-XPS-15"}
    ):
        result = await inventory_await_node(state)

    assert result["item_id"] == "IT-XPS-15"
    assert result["needs_clarification"] is False


# ── sourcing ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_quotes_resolves_supplier_by_email_not_name():
    """The core bug fix: supplier identity comes from the reply's sender email, never from
    fuzzy-matching the free-text company name an LLM parsed out of the PDF/email body."""
    from src.agents.workers.sourcing import extract_quotes

    mock_db = MagicMock()
    mock_db.get_all_suppliers.return_value = [
        {
            "supplier_id": "SUP-B",
            "name": "Global IT Supplies",
            "contact_email": "sales@globalit.com",
        },
    ]
    fake_reply = {
        "from": "Random Corp Sales <sales@globalit.com>",  # display name != supplier.name
        "subject": "Re: RFQ",
        "body_text": "Our price is RM 3,950.00, delivery 2 days, Net-60.",
        "attachments": [],
    }
    with (
        patch("src.agents.workers.sourcing.get_gmail_service", return_value=MagicMock()),
        patch("src.agents.workers.sourcing.fetch_replies", return_value=[fake_reply]),
        patch("src.agents.workers.sourcing.SupabaseRepository", return_value=mock_db),
        patch(
            "src.agents.workers.sourcing._parse_quotes_with_gemini",
            new=AsyncMock(return_value=[
                {
                    "supplier_name": "Some Totally Different Name Inc",
                    "unit_price_sen": 395000,
                    "quoted_delivery_days": 2,
                    "payment_terms": "Net-60",
                },
            ]),
        ),
    ):
        result = await extract_quotes(["sales@globalit.com"], "2026-07-01T00:00:00Z")

    assert len(result["extracted_quotes"]) == 1
    quote = result["extracted_quotes"][0]
    assert quote["supplier_id"] == "SUP-B"
    assert quote["supplier_name"] == "Global IT Supplies"  # DB name, not the mismatched parsed name


@pytest.mark.asyncio
async def test_extract_quotes_unknown_sender_tagged_unknown():
    from src.agents.workers.sourcing import extract_quotes

    mock_db = MagicMock()
    mock_db.get_all_suppliers.return_value = [
        {
            "supplier_id": "SUP-B",
            "name": "Global IT Supplies",
            "contact_email": "sales@globalit.com",
        },
    ]
    fake_reply = {
        "from": "nobody@unknown.com",
        "subject": "x",
        "body_text": "RM 1,000, 5 days, Net-30",
        "attachments": [],
    }
    with (
        patch("src.agents.workers.sourcing.get_gmail_service", return_value=MagicMock()),
        patch("src.agents.workers.sourcing.fetch_replies", return_value=[fake_reply]),
        patch("src.agents.workers.sourcing.SupabaseRepository", return_value=mock_db),
        patch(
            "src.agents.workers.sourcing._parse_quotes_with_gemini",
            new=AsyncMock(return_value=[
                {
                    "supplier_name": "Whoever",
                    "unit_price_sen": 100000,
                    "quoted_delivery_days": 5,
                    "payment_terms": "Net-30",
                },
            ]),
        ),
    ):
        result = await extract_quotes(["sales@globalit.com"], "2026-07-01T00:00:00Z")

    assert result["extracted_quotes"][0]["supplier_id"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_wait_for_quotes_all_replied():
    from src.agents.workers.sourcing import wait_for_quotes

    with (
        patch("src.agents.workers.sourcing.get_gmail_service", return_value=MagicMock()),
        patch(
            "src.agents.workers.sourcing.fetch_replies",
            return_value=[{"from": "a@b.com", "subject": "", "body_text": "", "attachments": []}],
        ),
    ):
        result = await wait_for_quotes(["a@b.com"], "2026-07-01T00:00:00Z")

    assert result == {"all_replied": True, "pending_emails": []}


@pytest.mark.asyncio
async def test_wait_for_quotes_timeout_returns_pending():
    from src.agents.workers.sourcing import wait_for_quotes

    with (
        patch("src.agents.workers.sourcing.get_gmail_service", return_value=MagicMock()),
        patch("src.agents.workers.sourcing.fetch_replies", return_value=[]),
    ):
        result = await wait_for_quotes(
            ["a@b.com", "c@d.com"], "2026-07-01T00:00:00Z", timeout_seconds=0
        )

    assert result["all_replied"] is False
    assert set(result["pending_emails"]) == {"a@b.com", "c@d.com"}


@pytest.mark.asyncio
async def test_sourcing_node_sends_rfq_when_not_sent_yet():
    from src.agents.workers.sourcing import sourcing_node

    state: ProcurementState = {"session_id": "s1", "item_name": "Dell XPS 15", "requested_qty": 30}
    fake_send = AsyncMock(
        return_value={"rfq_sent_at": "2026-07-01T00:00:00Z", "supplier_emails": ["a@b.com"]}
    )
    with patch("src.agents.workers.sourcing.send_rfqs", fake_send):
        result = await sourcing_node(state)

    assert result["rfq_sent_at"] == "2026-07-01T00:00:00Z"
    assert result["supplier_emails"] == ["a@b.com"]


@pytest.mark.asyncio
async def test_sourcing_node_extracts_quotes_once_all_replied():
    from src.agents.workers.sourcing import sourcing_node

    state: ProcurementState = {
        "session_id": "s1",
        "rfq_sent_at": "t",
        "supplier_emails": ["a@b.com"],
    }
    fake_wait = AsyncMock(return_value={"all_replied": True, "pending_emails": []})
    fake_extract = AsyncMock(return_value={"extracted_quotes": [{"supplier_id": "SUP-B"}]})
    with (
        patch("src.agents.workers.sourcing.wait_for_quotes", fake_wait),
        patch("src.agents.workers.sourcing.extract_quotes", fake_extract),
    ):
        result = await sourcing_node(state)

    assert result["extracted_quotes"] == [{"supplier_id": "SUP-B"}]
    assert result["all_replied"] is True


@pytest.mark.asyncio
async def test_sourcing_node_asks_user_on_partial_timeout():
    from src.agents.workers.sourcing import sourcing_node

    state: ProcurementState = {
        "session_id": "s1",
        "rfq_sent_at": "t",
        "supplier_emails": ["a@b.com", "c@d.com"],
    }
    fake_wait = AsyncMock(return_value={"all_replied": False, "pending_emails": ["c@d.com"]})
    with patch("src.agents.workers.sourcing.wait_for_quotes", fake_wait):
        result = await sourcing_node(state)

    assert result["needs_clarification"] is True
    assert result["clarification_payload"]["pending_emails"] == ["c@d.com"]
    assert result["clarification_payload"]["options"] == [
        "proceed_partial",
        "extend_wait",
        "send_reminder",
    ]


@pytest.mark.asyncio
async def test_sourcing_node_catches_exceptions():
    from src.agents.workers.sourcing import sourcing_node

    state: ProcurementState = {"session_id": "s1", "item_name": "x", "requested_qty": 1}
    with patch(
        "src.agents.workers.sourcing.send_rfqs", AsyncMock(side_effect=RuntimeError("no suppliers"))
    ):
        result = await sourcing_node(state)

    assert result["error"] == "no suppliers"


@pytest.mark.asyncio
async def test_sourcing_await_node_passthrough_when_not_needed():
    from src.agents.workers.sourcing import sourcing_await_node

    state: ProcurementState = {"session_id": "s1", "needs_clarification": False}
    result = await sourcing_await_node(state)
    assert result == state


@pytest.mark.asyncio
async def test_sourcing_await_node_proceed_partial_extracts_quotes():
    from src.agents.workers.sourcing import sourcing_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "supplier_emails": ["a@b.com"],
        "rfq_sent_at": "t",
        "pending_emails": ["c@d.com"],
        "needs_clarification": True,
        "clarification_payload": {"type": "sourcing_timeout"},
    }
    fake_extract = AsyncMock(return_value={"extracted_quotes": [{"supplier_id": "SUP-B"}]})
    with (
        patch("src.agents.workers.sourcing.interrupt", return_value={"action": "proceed_partial"}),
        patch("src.agents.workers.sourcing.extract_quotes", fake_extract),
    ):
        result = await sourcing_await_node(state)

    assert result["extracted_quotes"] == [{"supplier_id": "SUP-B"}]
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_sourcing_await_node_extend_wait_just_clears_flag():
    from src.agents.workers.sourcing import sourcing_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "needs_clarification": True,
        "clarification_payload": {"type": "sourcing_timeout"},
    }
    with patch("src.agents.workers.sourcing.interrupt", return_value={"action": "extend_wait"}):
        result = await sourcing_await_node(state)

    assert result["needs_clarification"] is False
    assert "extracted_quotes" not in result


@pytest.mark.asyncio
async def test_sourcing_await_node_send_reminder_then_waits_again():
    from src.agents.workers.sourcing import sourcing_await_node

    state: ProcurementState = {
        "session_id": "s1",
        "pending_emails": ["c@d.com"],
        "needs_clarification": True,
        "clarification_payload": {"type": "sourcing_timeout"},
    }
    fake_reminder = AsyncMock(return_value={"reminded_emails": ["c@d.com"]})
    with (
        patch("src.agents.workers.sourcing.interrupt", return_value={"action": "send_reminder"}),
        patch("src.agents.workers.sourcing.send_reminder_email", fake_reminder),
    ):
        result = await sourcing_await_node(state)

    fake_reminder.assert_awaited_once_with(["c@d.com"])
    assert result["needs_clarification"] is False


# ── evaluation ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_purchase_history_tool_wraps_query_history():
    from src.agents.workers.evaluation import get_purchase_history

    mock_db = MagicMock()
    mock_db.get_purchase_history.return_value = [{"unit_price_sen": 365000, "delivery_days": 7}]
    with patch("src.agents.tools.history.SupabaseRepository", return_value=mock_db):
        result = await get_purchase_history.ainvoke({"item_id": "IT-XPS-15"})

    assert result["avg_unit_price_sen"] == 365000.0


def test_get_reference_score_tool_wraps_score_suppliers():
    from src.agents.workers.evaluation import get_reference_score

    quotes = [
        {
            "supplier_id": "SUP-A",
            "unit_price_sen": 410000,
            "quoted_delivery_days": 5,
            "payment_terms": "Net-30",
        },
        {
            "supplier_id": "SUP-B",
            "unit_price_sen": 395000,
            "quoted_delivery_days": 2,
            "payment_terms": "Net-60",
        },
    ]
    result = get_reference_score.invoke(
        {"quotes": quotes, "avg_unit_price_sen": 365000.0, "avg_delivery_days": 7.0}
    )
    recommended = next(r for r in result if r["is_recommended"])
    assert recommended["supplier_id"] == "SUP-B"


def test_write_audit_log_tool_records_decision():
    from src.agents.workers.evaluation import write_audit_log

    def _supplier(supplier_id: str, is_recommended: bool) -> dict:
        return {
            "supplier_id": supplier_id,
            "supplier_name": f"Supplier {supplier_id}",
            "unit_price_sen": 100000,
            "quoted_delivery_days": 7,
            "payment_terms": "Net-30",
            "total_score": 80.0,
            "risk_flags": [],
            "is_recommended": is_recommended,
            "reasoning": "test",
        }

    mock_db = MagicMock()
    evaluated = [_supplier("SUP-A", False), _supplier("SUP-B", True)]
    with patch("src.agents.workers.evaluation.SupabaseRepository", return_value=mock_db):
        write_audit_log.invoke(
            {"evaluated_suppliers": evaluated, "overall_reasoning": "SUP-B is cheaper and faster."}
        )

    mock_db.write_audit_log.assert_called_once()
    _, kwargs = mock_db.write_audit_log.call_args
    assert kwargs["action_type"] == "SUPPLIER_EVALUATION"
    assert kwargs["decision_json"]["recommended_supplier_id"] == "SUP-B"
    assert kwargs["decision_json"]["overall_reasoning"] == "SUP-B is cheaper and faster."


@pytest.mark.asyncio
async def test_evaluation_node_writes_evaluated_suppliers_from_audit_log_call(fake_llm):
    from src.agents.workers.evaluation import evaluation_node

    evaluated = [
        {
            "supplier_id": "SUP-B",
            "supplier_name": "Global IT",
            "unit_price_sen": 395000,
            "quoted_delivery_days": 2,
            "payment_terms": "Net-60",
            "total_score": 92.0,
            "risk_flags": [],
            "is_recommended": True,
            "reasoning": "Best price and delivery, no risk flags.",
        }
    ]
    llm = fake_llm(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_audit_log",
                        "args": {
                            "evaluated_suppliers": evaluated,
                            "overall_reasoning": "SUP-B wins on price and speed.",
                        },
                        "id": "1",
                    }
                ],
            ),
        ]
    )
    state: ProcurementState = {"session_id": "s1", "item_id": "IT-XPS-15", "extracted_quotes": []}
    with (
        patch("src.agents.workers.evaluation._build_llm", return_value=llm),
        patch("src.agents.workers.evaluation.SupabaseRepository", return_value=MagicMock()),
    ):
        result = await evaluation_node(state)

    assert result["evaluated_suppliers"] == evaluated
    assert "error" not in result


@pytest.mark.asyncio
async def test_evaluation_node_fails_if_agent_skips_audit_log(fake_llm):
    from src.agents.workers.evaluation import evaluation_node

    llm = fake_llm([AIMessage(content="SUP-B looks best.")])  # no tool call
    state: ProcurementState = {"session_id": "s1", "item_id": "IT-XPS-15", "extracted_quotes": []}
    with patch("src.agents.workers.evaluation._build_llm", return_value=llm):
        result = await evaluation_node(state)

    assert "error" in result


@pytest.mark.asyncio
async def test_evaluation_node_catches_exceptions():
    from src.agents.workers.evaluation import evaluation_node

    state: ProcurementState = {"session_id": "s1", "item_id": "IT-XPS-15", "extracted_quotes": []}
    with patch("src.agents.workers.evaluation._build_llm", side_effect=RuntimeError("boom")):
        result = await evaluation_node(state)

    assert result["error"] == "boom"


# ── reporting ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reporting_node_assembles_report_with_summary():
    from src.agents.workers.reporting import reporting_node

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Global IT offers the best deal."))
    state: ProcurementState = {
        "session_id": "s1",
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
                "total_score": 95.0,
                "risk_flags": [],
                "is_recommended": True,
                "reasoning": "Cheapest and fastest.",
            }
        ],
    }
    with patch("src.agents.workers.reporting._build_llm", return_value=mock_llm):
        result = await reporting_node(state)

    assert "Global IT offers the best deal." in result["report_markdown"]
    assert "Global IT" in result["report_markdown"]
    assert "error" not in result


@pytest.mark.asyncio
async def test_reporting_node_handles_no_evaluation_needed():
    from src.agents.workers.reporting import reporting_node

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Stock is sufficient, no purchase needed.")
    )
    state: ProcurementState = {
        "session_id": "s1",
        "item_name": "Ergonomic Office Chair",
        "requested_qty": 5,
        "stock_sufficient": True,
        "current_stock": 30,
        "evaluated_suppliers": [],
    }
    with patch("src.agents.workers.reporting._build_llm", return_value=mock_llm):
        result = await reporting_node(state)

    assert "error" not in result
    assert "Ergonomic Office Chair" in result["report_markdown"]


@pytest.mark.asyncio
async def test_reporting_node_catches_exceptions():
    from src.agents.workers.reporting import reporting_node

    state: ProcurementState = {"session_id": "s1", "item_name": "x", "requested_qty": 1}
    with patch("src.agents.workers.reporting._build_llm", side_effect=RuntimeError("boom")):
        result = await reporting_node(state)

    assert result["error"] == "boom"


@pytest.mark.asyncio
async def test_reporting_node_handles_gemini_3_list_content():
    """Regression: Gemini 3 'thinking' models return AIMessage.content as a list of parts
    (each carrying a thought_signature), not a plain string — a live run against the real API
    showed the raw repr of that list literally rendered into the report before this was fixed."""
    from src.agents.workers.reporting import reporting_node

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=[
                {"type": "text", "text": "Stock is sufficient, no purchase needed.", "extras": {"signature": "abc"}}
            ]
        )
    )
    state: ProcurementState = {
        "session_id": "s1",
        "item_name": "Ergonomic Office Chair",
        "requested_qty": 5,
        "stock_sufficient": True,
        "current_stock": 30,
        "evaluated_suppliers": [],
    }
    with patch("src.agents.workers.reporting._build_llm", return_value=mock_llm):
        result = await reporting_node(state)

    assert "Stock is sufficient, no purchase needed." in result["report_markdown"]
    assert "extras" not in result["report_markdown"]
    assert "signature" not in result["report_markdown"]
