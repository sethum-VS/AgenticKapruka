"""Unit tests for graphs.nodes.analyze_intent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graphs.nodes.analyze_intent import (
    PROCEED_CHECKOUT_MESSAGE,
    IntentClassification,
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
async def test_analyze_intent_discovery_for_birthday_cake() -> None:
    """Mocked Gemini client returns discovery for a gift search message."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = IntentClassification(intent="discovery")
    mock_response.text = '{"intent": "discovery"}'
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "session_id": "sess-intent-001",
    }

    with patch(
        "graphs.nodes.analyze_intent.select_intent_model",
        return_value="gemini-2.5-flash",
    ):
        result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("birthday cake for mom")
    assert result == {"intent": "discovery", "intent_metadata": expected_metadata}
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "birthday cake for mom"
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is IntentClassification


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
async def test_analyze_intent_parses_json_text_when_parsed_missing() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = '{"intent": "tracking"}'
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="where is order VIMP123?")],
        "session_id": "sess-intent-004",
    }

    with patch(
        "graphs.nodes.analyze_intent.select_intent_model",
        return_value="gemini-2.5-flash",
    ):
        result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("where is order VIMP123?")
    assert result == {"intent": "tracking", "intent_metadata": expected_metadata}


@pytest.mark.asyncio
async def test_analyze_intent_uses_lora_endpoint_when_configured() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = IntentClassification(intent="discovery")
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="avurudu cake ona")],
        "session_id": "sess-intent-lora",
    }
    lora_model = "projects/test/locations/us-central1/endpoints/lora-1"

    with patch(
        "graphs.nodes.analyze_intent.select_intent_model",
        return_value=lora_model,
    ):
        result = await analyze_intent(state, genai_client=mock_client)

    expected_metadata: IntentMetadata = _preprocessor.process("avurudu cake ona")
    assert result == {"intent": "discovery", "intent_metadata": expected_metadata}
    assert mock_client.models.generate_content.call_args.kwargs["model"] == lora_model


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
async def test_analyze_intent_sets_delivery_metadata_before_llm() -> None:
    """Preprocessor runs on raw message and stores delivery city before Gemini."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = IntentClassification(intent="discovery")
    mock_client.models.generate_content.return_value = mock_response

    message = "Machan, can you deliver to Kandy on Sunday?"
    state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "session_id": "sess-intent-delivery",
    }

    with patch(
        "graphs.nodes.analyze_intent.select_intent_model",
        return_value="gemini-2.5-flash",
    ):
        result = await analyze_intent(state, genai_client=mock_client)

    metadata = result["intent_metadata"]
    assert metadata is not None
    assert metadata["detected_vernacular"] == "tanglish"
    assert metadata["requires_delivery_validation"] is True
    assert metadata["target_city"] == "Kandy"
    mock_client.models.generate_content.assert_called_once()
