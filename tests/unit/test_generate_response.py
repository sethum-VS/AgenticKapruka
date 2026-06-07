"""Unit tests for graphs.nodes.generate_response."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.generate_response import (
    AssistantReply,
    generate_response,
    render_assistant_html,
)
from graphs.state import AgentState
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL

_SEARCH_TOOL_RESULTS = {
    SEARCH_PRODUCTS_TOOL: {
        "results": [
            {
                "id": "cake00ka002034",
                "name": "Chocolate Birthday Cake",
                "summary": "Rich chocolate layers.",
                "price": {"amount": 4500.0, "currency": "LKR"},
                "compare_at_price": None,
                "in_stock": True,
                "stock_level": "high",
                "image_url": "https://example.com/cake.jpg",
                "category": {
                    "id": "cat_cakes",
                    "name": "Birthday",
                    "slug": "birthday",
                },
                "rating": None,
                "ships_internationally": False,
                "url": "https://www.kapruka.com/cake",
            },
            {
                "id": "cake00ka002099",
                "name": "Vanilla Celebration Cake",
                "summary": "Classic vanilla sponge.",
                "price": {"amount": 3800.0, "currency": "LKR"},
                "compare_at_price": None,
                "in_stock": True,
                "stock_level": "medium",
                "image_url": None,
                "category": {
                    "id": "cat_cakes",
                    "name": "Birthday",
                    "slug": "birthday",
                },
                "rating": None,
                "ships_internationally": False,
                "url": "https://www.kapruka.com/vanilla-cake",
            },
        ],
        "next_cursor": None,
        "applied_filters": {
            "q": "birthday cake for mom",
            "limit": 10,
            "in_stock_only": False,
        },
    },
}


@pytest.mark.asyncio
async def test_generate_response_html_contains_product_names_from_tool_results() -> None:
    """Mocked Gemini reply mentions MCP product names; partial HTML includes them."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message=(
            "I found two birthday cakes: Chocolate Birthday Cake (LKR 4,500) "
            "and Vanilla Celebration Cake (LKR 3,800)."
        ),
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "session_id": "sess-gen-001",
    }

    with patch("graphs.nodes.generate_response.FLASH_MODEL", "gemini-2.5-flash"):
        result = await generate_response(state, genai_client=mock_client)

    assert "response_html" in result
    html = result["response_html"]
    assert "Chocolate Birthday Cake" in html
    assert "Vanilla Celebration Cake" in html
    assert 'aria-label="Assistant message"' in html

    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert "tool_results" in call_kwargs["contents"]
    assert "Chocolate Birthday Cake" in call_kwargs["contents"]
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is AssistantReply


@pytest.mark.asyncio
async def test_generate_response_empty_user_message_skips_llm() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="   ")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "session_id": "sess-gen-002",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "response_html" in result
    assert "How can I help you" in result["response_html"]
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_parses_json_text_when_parsed_missing() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = '{"message": "Chocolate Birthday Cake is in stock at LKR 4,500."}'
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="tell me about cakes")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "session_id": "sess-gen-003",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "Chocolate Birthday Cake" in result["response_html"]
    assert "LKR 4,500" in result["response_html"]


def test_render_assistant_html_structure() -> None:
    html = render_assistant_html("Hello from Kapruka!")
    assert "Hello from Kapruka!" in html
    assert 'role="article"' in html
    assert "justify-start" in html
