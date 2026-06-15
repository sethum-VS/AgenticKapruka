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
    assert result == {"intent": "discovery", "intent_metadata": expected_metadata}
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
async def test_analyze_intent_benchmark_eliminates_separate_intent_llm_call() -> None:
    """Phase 2: one fewer Gemini call — analyze_intent is guard-only for shopping turns."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="anniversary dinner gifts in Kandy")],
        "session_id": "sess-intent-benchmark",
    }

    await analyze_intent(state, genai_client=mock_client)

    assert mock_client.models.generate_content.call_count == 0
