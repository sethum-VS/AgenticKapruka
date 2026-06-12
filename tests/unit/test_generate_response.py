"""Unit tests for graphs.nodes.generate_response."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import HumanMessage

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.model_router import PRO_MODEL
from graphs.nodes.generate_response import (
    AssistantReply,
    _resolve_effective_tool_results,
    build_agent_tool_error_message,
    build_products_carousel_html,
    extract_search_products,
    generate_response,
    merge_tool_trace,
    render_assistant_html,
)
from graphs.state import AgentState, ToolInvocation
from lib.chat.delivery_dates import delivery_date_clarifying_question
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.system_prompts import (
    LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION,
    UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION,
    select_response_system_instruction,
)
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL

_CHECKOUT_REVIEW_HTML = '<section data-testid="checkout-review">Review summary</section>'

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

    result = await generate_response(state, genai_client=mock_client)

    assert "response_html" in result
    html = result["response_html"]
    assert "Chocolate Birthday Cake" in html
    assert "Vanilla Celebration Cake" in html
    assert 'aria-label="Assistant message"' in html
    assert 'data-slot="product-carousel"' in html
    assert 'data-testid="product-carousel"' in html
    assert 'data-product-id="cake00ka002034"' in html

    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert "tool_results" in call_kwargs["contents"]
    assert "Chocolate Birthday Cake" in call_kwargs["contents"]
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is AssistantReply


@pytest.mark.asyncio
async def test_generate_response_template_fallback_on_gemini_429() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = genai_errors.ClientError(
        429,
        {"error": {"status": "RESOURCE_EXHAUSTED"}},
        None,
    )

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "session_id": "sess-gen-429",
        "intent": "discovery",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "Chocolate Birthday Cake" in result["assistant_message"]
    assert "Vanilla Celebration Cake" in result["assistant_message"]
    assert 'data-testid="product-carousel"' in result["response_html"]


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
    assert 'role="assistant"' in html
    assert "prose-assistant" in html
    assert "justify-start" in html


def test_extract_search_products_from_tool_results() -> None:
    products = extract_search_products(_SEARCH_TOOL_RESULTS)
    assert len(products) == 2
    assert products[0]["id"] == "cake00ka002034"
    assert extract_search_products({}) == []
    assert extract_search_products(None) == []


def test_build_products_carousel_html_renders_carousel() -> None:
    html = build_products_carousel_html(_SEARCH_TOOL_RESULTS)
    assert html is not None
    assert 'data-testid="product-carousel"' in html
    assert "Chocolate Birthday Cake" in html


def test_build_products_carousel_html_empty_when_no_results() -> None:
    assert build_products_carousel_html({SEARCH_PRODUCTS_TOOL: {"results": []}}) is None


@pytest.mark.asyncio
async def test_generate_response_checkout_review_uses_pro_model_and_embeds_summary() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Your order looks ready. Please confirm the delivery details below.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="confirm my order")],
        "intent": "checkout",
        "checkout_state": "review",
        "model_tier": "pro",
        "tool_results": {
            CHECKOUT_TOOL_KEY: {
                "current_step": "review",
                "review_html": _CHECKOUT_REVIEW_HTML,
                "cart_items": [
                    {
                        "product_id": "cake00ka002034",
                        "name": "Chocolate Birthday Cake",
                        "quantity": 1,
                        "price_amount": 4500.0,
                    },
                ],
                "delivery_address": "123 Galle Road",
                "delivery_city": "Colombo 03",
                "recipient_name": "Ada",
                "recipient_phone": "0771234567",
                "sender_name": "Bob",
                "sender_anonymous": False,
            },
        },
        "session_id": "sess-gen-review-001",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert result["model_tier"] == "pro"
    assert 'data-slot="checkout-review"' in result["response_html"]
    assert 'data-testid="checkout-review"' in result["response_html"]
    assert "confirm" in result["assistant_message"].lower()

    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == PRO_MODEL
    assert "checkout_summary" in call_kwargs["contents"]


def test_select_response_system_instruction_utility_mode() -> None:
    metadata: IntentMetadata = {
        "is_situational": False,
        "detected_vernacular": "en",
        "requires_delivery_validation": False,
        "target_city": None,
    }
    prompt = select_response_system_instruction(metadata)
    assert prompt == UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION
    assert "transactional" in prompt.lower()
    assert "concierge" not in prompt.lower()


def test_select_response_system_instruction_situational_tanglish() -> None:
    metadata: IntentMetadata = {
        "is_situational": True,
        "detected_vernacular": "tanglish",
        "requires_delivery_validation": False,
        "target_city": None,
    }
    prompt = select_response_system_instruction(metadata)
    assert prompt.startswith(LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION)
    assert "Tanglish" in prompt
    assert "machan" in prompt


def test_select_response_system_instruction_defaults_to_utility() -> None:
    prompt = select_response_system_instruction(None)
    assert prompt == UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION


@pytest.mark.asyncio
async def test_generate_response_utility_metadata_uses_ecommerce_prompt() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="Chocolate Birthday Cake — LKR 4,500.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="show birthday cakes under 5000")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "en",
            "requires_delivery_validation": False,
            "target_city": None,
        },
        "session_id": "sess-gen-utility",
    }

    await generate_response(state, genai_client=mock_client)

    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert "transactional" in config.system_instruction.lower()


@pytest.mark.asyncio
async def test_generate_response_situational_metadata_uses_concierge_prompt() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Aiyo machan, here are gentle condolence flowers for her.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="mage girlfriend broke up, flowers ona")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "intent_metadata": {
            "is_situational": True,
            "detected_vernacular": "tanglish",
            "requires_delivery_validation": False,
            "target_city": None,
        },
        "session_id": "sess-gen-concierge",
    }

    await generate_response(state, genai_client=mock_client)

    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert "concierge" in config.system_instruction.lower()
    assert "Tanglish" in config.system_instruction


@pytest.mark.asyncio
async def test_generate_response_no_carousel_when_search_empty() -> None:
    mock_client = MagicMock()

    state: AgentState = {
        "messages": [HumanMessage(content="obscure cake query")],
        "intent": "discovery",
        "tool_results": {SEARCH_PRODUCTS_TOOL: {"results": []}},
        "session_id": "sess-gen-004",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert 'data-testid="product-carousel"' not in result["response_html"]
    assert "couldn't find products" in result["assistant_message"].lower()
    mock_client.models.generate_content.assert_not_called()


def _product(
    product_id: str,
    name: str,
    *,
    amount: float = 4500.0,
) -> dict[str, object]:
    return {
        "id": product_id,
        "name": name,
        "summary": f"Summary for {name}.",
        "price": {"amount": amount, "currency": "LKR"},
        "compare_at_price": None,
        "in_stock": True,
        "stock_level": "high",
        "image_url": None,
        "category": {"id": "cat_cakes", "name": "Birthday", "slug": "birthday"},
        "rating": None,
        "ships_internationally": False,
        "url": f"https://www.kapruka.com/{product_id}",
    }


def test_resolve_effective_tool_results_prefers_checkout_over_stale_trace() -> None:
    """Follow-up checkout turns must not reuse discovery tool_trace from checkpoint."""
    state: AgentState = {
        "intent": "checkout",
        "tool_trace": [
            {
                "name": "kapruka_search_products",
                "args": {"q": "birthday cake"},
                "result": {"results": []},
            },
        ],
        "tool_results": {
            CHECKOUT_TOOL_KEY: {
                "current_step": "delivery_city",
                "cart_items": [{"product_id": "cake00ka002034", "quantity": 1}],
            },
        },
    }
    resolved = _resolve_effective_tool_results(state)
    assert resolved is not None
    assert CHECKOUT_TOOL_KEY in resolved
    assert "kapruka_search_products" not in resolved


def test_merge_tool_trace_last_wins_non_search_tools() -> None:
    """Non-search tools keep the last invocation result per tool name."""
    trace: list[ToolInvocation] = [
        {
            "name": "kapruka_check_delivery",
            "args": {"city": "Kandy"},
            "result": {"city": "Kandy", "deliverable": False},
        },
        {
            "name": "kapruka_check_delivery",
            "args": {"city": "Colombo"},
            "result": {"city": "Colombo", "deliverable": True},
        },
    ]
    merged = merge_tool_trace(trace)
    assert merged["kapruka_check_delivery"]["city"] == "Colombo"
    assert merged["kapruka_check_delivery"]["deliverable"] is True


def test_merge_tool_trace_unions_search_product_ids() -> None:
    """Search invocations merge into one payload with deduped product union."""
    trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "chocolate cake"},
            "result": {
                "results": [_product("cake00ka002034", "Chocolate Birthday Cake")],
                "applied_filters": {"q": "chocolate cake"},
            },
        },
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "vanilla cake"},
            "result": {
                "results": [
                    _product("cake00ka002034", "Chocolate Birthday Cake"),
                    _product("cake00ka002099", "Vanilla Celebration Cake"),
                ],
                "applied_filters": {"q": "vanilla cake"},
            },
        },
    ]
    merged = merge_tool_trace(trace)
    search_payload = merged[SEARCH_PRODUCTS_TOOL]
    assert search_payload["applied_filters"] == {"q": "vanilla cake"}
    product_ids = [item["id"] for item in search_payload["results"]]
    assert product_ids == ["cake00ka002034", "cake00ka002099"]


@pytest.mark.asyncio
async def test_generate_response_merged_trace_carousel_union() -> None:
    """Multi-step agent loop trace renders carousel from unioned search products."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Here are cakes from both searches.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "chocolate cake"},
            "result": {
                "results": [_product("cake00ka002034", "Chocolate Birthday Cake")],
            },
        },
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "vanilla cake"},
            "result": {
                "results": [_product("cake00ka002099", "Vanilla Celebration Cake")],
            },
        },
    ]

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cakes for mom")],
        "tool_trace": tool_trace,
        "agent_loop_done": True,
        "session_id": "sess-gen-trace-union",
    }

    result = await generate_response(state, genai_client=mock_client)

    html = result["response_html"]
    assert 'data-testid="product-carousel"' in html
    assert 'data-product-id="cake00ka002034"' in html
    assert 'data-product-id="cake00ka002099"' in html
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "Chocolate Birthday Cake" in call_kwargs["contents"]
    assert "Vanilla Celebration Cake" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_generate_response_clarifying_question_skips_gemini() -> None:
    """ask_user exit renders clarifying question without catalog synthesis."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="send flowers to my aunt")],
        "intent": "discovery",
        "agent_clarifying_question": "Which city should we deliver to?",
        "agent_loop_exit_reason": "ask_user",
        "tool_trace": [
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "flowers"},
                "result": {"results": [_product("flw001", "Rose Bouquet")]},
            },
        ],
        "session_id": "sess-gen-clarify",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert result["assistant_message"] == "Which city should we deliver to?"
    assert "Which city should we deliver to?" in result["response_html"]
    assert 'data-testid="product-carousel"' not in result["response_html"]
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_ignores_stale_clarifying_question_on_finish() -> None:
    """Stale ask_user clarifying text must not mask fresh search products on finish."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="Here are some cakes from Kapruka.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "cakes"},
            "result": {
                "results": [_product("cake00ka002034", "Chocolate Birthday Cake")],
            },
        },
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "edible cakes"},
            "result": {
                "results": [_product("cake00ka002099", "Vanilla Celebration Cake")],
            },
        },
    ]

    state: AgentState = {
        "messages": [HumanMessage(content="cakes")],
        "intent": "discovery",
        "agent_clarifying_question": "The previous search for 'gifts' returned no products.",
        "agent_loop_exit_reason": "finish",
        "agent_loop_done": True,
        "tool_trace": tool_trace,
        "session_id": "sess-gen-stale-clarify",
    }

    result = await generate_response(state, genai_client=mock_client)

    html = result["response_html"]
    assert "The previous search for 'gifts'" not in result["assistant_message"]
    assert 'data-testid="product-carousel"' in html
    assert 'data-product-id="cake00ka002034"' in html
    assert 'data-product-id="cake00ka002099"' in html
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_empty_merged_search_fallback_via_tool_trace() -> None:
    """Empty unioned search in tool_trace keeps empathetic discovery fallback."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="obscure gift idea")],
        "intent": "discovery",
        "tool_trace": [
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "first try"},
                "result": {"results": []},
            },
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "second try"},
                "result": {"results": []},
            },
        ],
        "agent_loop_done": True,
        "session_id": "sess-gen-trace-empty",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "couldn't find products" in result["assistant_message"].lower()
    assert 'data-testid="product-carousel"' not in result["response_html"]
    mock_client.models.generate_content.assert_not_called()


def test_build_agent_tool_error_message_past_delivery_date() -> None:
    """Past delivery MCP errors prompt for a valid delivery date."""
    message = build_agent_tool_error_message(
        tool=CHECK_DELIVERY_TOOL,
        raw_message="Choose a delivery date that is today or later.",
        error_code="past_delivery_date",
    )
    assert message == delivery_date_clarifying_question()


def test_build_agent_tool_error_message_generic_mcp() -> None:
    """Generic MCP failures use problem + cause + fix copy."""
    message = build_agent_tool_error_message(
        tool=SEARCH_PRODUCTS_TOOL,
        raw_message="Kapruka search timed out.",
        error_code="upstream_error",
    )
    assert "could not search the kapruka catalog" in message.lower()
    assert "Kapruka search timed out." in message
    assert "adjust your request" in message.lower()


@pytest.mark.asyncio
async def test_generate_response_tool_error_past_delivery_skips_gemini() -> None:
    """tool_error exit renders delivery-date guidance without catalog synthesis."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="deliver cake to Colombo on 2024-01-01")],
        "intent": "discovery",
        "agent_loop_exit_reason": "tool_error",
        "agent_tool_error": {
            "tool": CHECK_DELIVERY_TOOL,
            "message": "Choose a delivery date that is today or later.",
        },
        "tool_trace": [
            {
                "name": CHECK_DELIVERY_TOOL,
                "args": {"city": "Colombo 03", "delivery_date": "2024-01-01"},
                "result": {
                    "error": "past_delivery_date",
                    "message": "Choose a delivery date that is today or later.",
                },
            },
        ],
        "session_id": "sess-gen-tool-error-delivery",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert result["assistant_message"] == delivery_date_clarifying_question()
    assert "When would you like delivery?" in result["response_html"]
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_tool_error_generic_mcp_skips_gemini() -> None:
    """tool_error exit renders tier-1 MCP failure copy without Gemini."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cakes")],
        "intent": "discovery",
        "agent_loop_exit_reason": "tool_error",
        "agent_tool_error": {
            "tool": SEARCH_PRODUCTS_TOOL,
            "message": "Kapruka search timed out.",
        },
        "tool_trace": [
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "birthday cake"},
                "result": {
                    "error": "upstream_error",
                    "message": "Kapruka search timed out.",
                },
            },
        ],
        "session_id": "sess-gen-tool-error-search",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "could not search the kapruka catalog" in result["assistant_message"].lower()
    assert "Kapruka search timed out." in result["assistant_message"]
    assert 'data-testid="product-carousel"' not in result["response_html"]
    mock_client.models.generate_content.assert_not_called()
