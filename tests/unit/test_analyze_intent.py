"""Unit tests for graphs.nodes.analyze_intent guard routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graphs.nodes.analyze_intent import (
    PROCEED_CHECKOUT_MESSAGE,
    _extract_latest_user_message,
    analyze_intent,
)
from graphs.state import AgentState
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.query_preprocessor import QueryPreprocessor

_preprocessor = QueryPreprocessor()


def test_extract_latest_user_message_prefers_last_human() -> None:
    messages = [
        HumanMessage(content="first question"),
        AIMessage(content="assistant reply"),
        HumanMessage(content="birthday cake for mom"),
    ]
    assert _extract_latest_user_message(messages) == "birthday cake for mom"


@pytest.mark.asyncio
async def test_analyze_intent_shopping_turn_skips_gemini() -> None:
    """Shopping turns default to discovery and defer refinement to agent_loop."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "session_id": "sess-intent-001",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("birthday cake for mom")
    assert result == {
        "intent": "discovery",
        "intent_metadata": expected_metadata,
        "session_product_focus": "cake",
        "session_occasion": "birthday",
        "session_recipient_hint": "mom",
    }
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_empty_message_defaults_to_general() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="   ")],
        "session_id": "sess-intent-002",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("   ")
    assert result == {"intent": "general", "intent_metadata": expected_metadata}
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_no_messages_defaults_to_general() -> None:
    mock_client = MagicMock()
    state: AgentState = {"messages": [], "session_id": "sess-intent-003"}

    result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("")
    assert result == {"intent": "general", "intent_metadata": expected_metadata}
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_tracking_guard_skips_gemini() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="where is order VIMP123?")],
        "session_id": "sess-intent-004",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("where is order VIMP123?")
    assert result == {"intent": "tracking", "intent_metadata": expected_metadata}
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_proceed_checkout_skips_gemini() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content=PROCEED_CHECKOUT_MESSAGE)],
        "session_id": "sess-intent-005",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process(PROCEED_CHECKOUT_MESSAGE)
    assert result == {"intent": "checkout", "intent_metadata": expected_metadata}
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_cart_add_routes_to_cart_intent() -> None:
    mock_client = MagicMock()
    message = "Add the Blush Roses combo to my cart please"
    state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "session_id": "sess-intent-cart-add",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["intent"] == "cart"
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_checkout_trigger_skips_gemini() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="I want to checkout my cart")],
        "session_id": "sess-intent-checkout",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["intent"] == "checkout"
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_delivery_question_routes_to_shopping_path() -> None:
    """Discovery delivery questions must not hit checkout guard — defer to agent_loop."""
    mock_client = MagicMock()
    message = "Machan, can you deliver to Kandy on Sunday?"
    state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "session_id": "sess-intent-delivery",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    metadata = result["intent_metadata"]
    assert result["intent"] == "discovery"
    assert metadata is not None
    assert metadata["detected_vernacular"] == "tanglish"
    assert metadata["requires_delivery_validation"] is True
    assert metadata["target_city"] == "Kandy"
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_intent_persists_session_budget_max() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="cakes under 5000 rupees")],
        "session_id": "sess-intent-budget",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["session_budget_max"] == 5000.0
    assert result["intent_metadata"]["budget_max"] == 5000.0


@pytest.mark.asyncio
async def test_analyze_intent_keeps_prior_session_budget_when_turn_has_none() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="show me flowers")],
        "session_id": "sess-intent-budget-carry",
        "session_budget_max": 8000.0,
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["session_budget_max"] == 8000.0
    assert result["intent_metadata"]["budget_max"] is None


@pytest.mark.asyncio
async def test_analyze_intent_persists_session_product_focus() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "session_id": "sess-intent-focus",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["session_product_focus"] == "cake"


@pytest.mark.asyncio
async def test_analyze_intent_keeps_prior_product_focus_on_floral_follow_up() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="She loves floral designs")],
        "session_id": "sess-intent-focus-carry",
        "session_product_focus": "cake",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["session_product_focus"] == "cake"


@pytest.mark.asyncio
async def test_analyze_intent_rehydrates_session_delivery_date() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Colombo 03 please")],
        "session_id": "sess-intent-date",
        "session_delivery_date": "2026-06-21",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["delivery_date"] == "2026-06-21"
    assert result["session_delivery_date"] == "2026-06-21"


@pytest.mark.asyncio
async def test_analyze_intent_vague_gift_asks_preferences() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="any gift ideas?")],
        "session_id": "sess-vague-gift",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result.get("agent_clarifying_question")
    assert result.get("session_awaiting_gift_preferences") is True


@pytest.mark.asyncio
async def test_analyze_intent_budgeted_gift_chip_routes_to_search() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Gift ideas under Rs. 5,000")],
        "session_id": "sess-budget-chip",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result.get("agent_clarifying_question") is None
    assert result.get("session_awaiting_gift_preferences") is not True
    assert result["intent_metadata"].get("budgeted_gift_discovery") is True


@pytest.mark.asyncio
async def test_analyze_intent_usd_session_rs_chip_uses_lkr_budget() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Gift ideas under Rs. 5,000")],
        "session_id": "sess-budget-currency",
        "currency": "USD",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["session_budget_max"] == 5000.0
    assert result["session_budget_currency"] == "LKR"


@pytest.mark.asyncio
async def test_analyze_intent_benchmark_eliminates_separate_intent_llm_call() -> None:
    """Phase 2: one fewer Gemini call — analyze_intent is guard-only for shopping turns."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="anniversary dinner gifts in Kandy")],
        "session_id": "sess-intent-benchmark",
    }

    await analyze_intent(state, genai_client=mock_client)

    assert mock_client.models.generate_content.call_count == 0


@pytest.mark.asyncio
async def test_analyze_intent_weather_routes_general_off_topic() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="What's the weather in Colombo?")],
        "session_id": "sess-off-topic",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result["intent"] == "general"
    assert result["intent_metadata"]["is_off_topic"] is True
    assert result["intent_metadata"]["target_city"] is None


@pytest.mark.asyncio
async def test_analyze_intent_topic_pivot_clears_session_budget() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Nevermind. Cakes.")],
        "session_id": "sess-pivot-budget",
        "session_budget_max": 6000.0,
        "session_budget_currency": "LKR",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result.get("session_budget_max") is None
    assert result.get("last_visible_products") is None
    assert result.get("last_search_products") is None


@pytest.mark.asyncio
async def test_analyze_intent_topic_pivot_clears_hybrid_hints_and_search_query() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Nevermind. Cakes.")],
        "session_id": "sess-pivot-context",
        "session_search_query": "anniversary gift hamper",
        "hybrid_context": {
            "hints": {"occasion": "Anniversary", "category": "Birthday"},
            "occasions": ["Anniversary"],
        },
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result.get("session_search_query") is None
    assert result.get("last_visible_products") is None
    assert result.get("last_search_products") is None
    assert result["intent_metadata"]["topic_pivot"] is True
    hints = result["hybrid_context"]["hints"]
    assert "occasion" not in hints
    assert "category" not in hints


@pytest.mark.asyncio
async def test_analyze_intent_occasion_change_sets_budget_confirmation_pending() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Show me some anniversary gifts")],
        "session_id": "sess-occasion-pivot",
        "session_occasion": "birthday",
        "session_budget_max": 6000.0,
        "session_budget_currency": "LKR",
    }

    result = await analyze_intent(state, genai_client=mock_client)

    assert result.get("session_occasion") == "anniversary"
    assert result.get("session_budget_max") == 6000.0
    assert result["intent_metadata"].get("budget_confirmation_pending") is True
