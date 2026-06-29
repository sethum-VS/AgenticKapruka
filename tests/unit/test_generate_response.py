"""Unit tests for graphs.nodes.generate_response."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import HumanMessage

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.model_router import PRO_MODEL
from graphs.nodes.generate_response import (
    AssistantReply,
    _apply_perishable_delivery_honesty,
    _build_discovery_template_reply,
    _build_user_prompt,
    _build_verified_city_delivery_line,
    _build_verified_delivery_fee_line,
    _cap_search_products_for_llm_context,
    _carousel_strict_budget,
    _format_product_line,
    _resolve_effective_tool_results,
    build_agent_tool_error_message,
    build_products_carousel_html,
    carousel_consistency_guard,
    delivery_claim_guard,
    extract_search_products,
    generate_response,
    merge_tool_trace,
    stock_consistency_guard,
    render_assistant_html,
)
from graphs.state import AgentState, ToolInvocation
from lib.chat.delivery_dates import delivery_date_clarifying_question
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.system_prompts import (
    GENERAL_TOOL_RESULTS_SYSTEM_INSTRUCTION,
    LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION,
    UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION,
    build_farewell_message,
    build_general_welcome_message,
    is_farewell_message,
    select_response_system_instruction,
)
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL

_CHECKOUT_REVIEW_HTML = '<section data-testid="checkout-review">Review summary</section>'


def _combined_response_html(result: dict[str, object]) -> str:
    """Merge split SSE response_html and carousel_html for assertions."""
    parts = [str(result.get("response_html") or "")]
    carousel = result.get("carousel_html")
    if isinstance(carousel, str):
        parts.append(carousel)
    return "".join(parts)

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
    html = _combined_response_html(result)
    assert "Chocolate Birthday Cake" in html
    assert "Vanilla Celebration Cake" in html
    assert 'aria-label="Assistant message"' in html
    assert 'data-slot="product-carousel"' in result["response_html"]
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
async def test_generate_response_decodes_html_entities_in_reply() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Try the Cadbury 135g &#8211; 30 Minis hamper for mom.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="chocolates for mom")],
        "tool_results": _SEARCH_TOOL_RESULTS,
        "session_id": "sess-gen-decode-001",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "Cadbury 135g – 30 Minis" in result["assistant_message"]
    assert "&#8211;" not in result["response_html"]


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

    assert "thoughtful Kapruka picks" in result["assistant_message"]
    assert "Chocolate Birthday Cake" in result["assistant_message"]
    assert "Vanilla Celebration Cake" in result["assistant_message"]
    assert 'data-testid="product-carousel"' in _combined_response_html(result)


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
    assert result["assistant_message"] == build_general_welcome_message()
    assert "Welcome to Kapruka" in result["response_html"]
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.parametrize(
    "user_message",
    [
        "hello",
        "what can you help with?",
    ],
)
@pytest.mark.asyncio
async def test_generate_response_general_welcome_skips_gemini(
    user_message: str,
) -> None:
    """General intent with no tools returns static welcome — no empty-catalog Gemini reply."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "intent": "general",
        "tool_trace": [],
        "agent_loop_exit_reason": "finish",
        "agent_loop_done": True,
        "session_id": f"sess-gen-welcome-{user_message[:8]}",
    }

    result = await generate_response(state, genai_client=mock_client)

    welcome = build_general_welcome_message()
    assert result["assistant_message"] == welcome
    assert "cakes" in result["assistant_message"].lower()
    assert "flowers" in result["assistant_message"].lower()
    assert "order tracking" in result["assistant_message"].lower()
    assert "couldn't find products" not in result["assistant_message"].lower()
    assert 'data-testid="product-carousel"' not in _combined_response_html(result)
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_support_policy_wilted_flowers_not_welcome() -> None:
    """Return/refund/quality questions get support handoff, not the welcome menu."""
    mock_client = MagicMock()
    user_message = "What's your return policy if flowers arrive wilted?"
    state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "intent": "general",
        "intent_metadata": cast(IntentMetadata, {"support_topic": "quality"}),
        "tool_trace": [],
        "session_id": "sess-support-faq",
    }

    result = await generate_response(state, genai_client=mock_client)

    reply = result["assistant_message"]
    welcome = build_general_welcome_message()
    assert reply != welcome
    assert "Welcome to Kapruka" not in reply
    assert "Kapruka support" in reply
    assert "+94-11-7551111" in reply
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_refined_general_farewell_uses_sign_off() -> None:
    """agent_loop refined_intent=general + finish with thanks that's all returns farewell."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="thanks that's all")],
        "intent": "general",
        "tool_trace": [],
        "tool_results": {},
        "agent_loop_exit_reason": "finish",
        "agent_loop_done": True,
        "agent_loop_iterations": 1,
        "session_id": "sess-gen-farewell",
    }

    result = await generate_response(state, genai_client=mock_client)

    farewell = build_farewell_message()
    assert result["assistant_message"] == farewell
    assert "Welcome to Kapruka" not in result["assistant_message"]
    assert "What would you like to explore" not in result["assistant_message"]
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.parametrize(
    "user_message",
    [
        "thanks!",
        "thank you",
        "that's all",
        "thanks, that's all",
        "goodbye",
    ],
)
def test_is_farewell_message(user_message: str) -> None:
    assert is_farewell_message(user_message)


def test_is_farewell_message_rejects_greeting() -> None:
    assert not is_farewell_message("hello")


@pytest.mark.asyncio
async def test_generate_response_general_thanks_returns_farewell() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="thanks!")],
        "intent": "general",
        "tool_trace": [],
        "agent_loop_exit_reason": "finish",
        "agent_loop_done": True,
        "session_id": "sess-gen-thanks-farewell",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert result["assistant_message"] == build_farewell_message()
    assert "Welcome to Kapruka" not in result["assistant_message"]
    mock_client.models.generate_content.assert_not_called()


def test_select_response_system_instruction_general_omits_empty_tool_results_rule() -> None:
    """General intent Gemini path must not instruct empty-catalog fallback copy."""
    prompt = select_response_system_instruction(None, intent="general")
    assert prompt == GENERAL_TOOL_RESULTS_SYSTEM_INSTRUCTION
    assert "tool_results are empty" not in prompt.lower()


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
    assert 'data-role="assistant-message"' in html


def test_extract_search_products_from_tool_results() -> None:
    products = extract_search_products(_SEARCH_TOOL_RESULTS)
    assert len(products) == 2
    assert products[0]["id"] == "cake00ka002034"
    assert extract_search_products({}) == []
    assert extract_search_products(None) == []


def test_extract_search_products_filters_cake_accessories() -> None:
    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                {
                    "id": "cake00ka002034",
                    "name": "Chocolate Birthday Cake",
                    "category": {"name": "Cakes", "slug": "cakes"},
                },
                {
                    "id": "acc001",
                    "name": "Gold Cake Topper",
                    "category": {"name": "Accessories", "slug": "accessories"},
                },
                {
                    "id": "acc002",
                    "name": "Silicone Cake Mould Set",
                    "category": {"name": "Baking", "slug": "baking"},
                },
                {
                    "id": "acc003",
                    "name": "Revolving Cake Turning Table",
                    "category": {"name": "Baking", "slug": "baking"},
                },
            ],
            "applied_filters": {"q": "cakes", "limit": 10},
        },
    }
    products = extract_search_products(tool_results)
    assert len(products) == 1
    assert products[0]["id"] == "cake00ka002034"


def test_extract_search_products_keeps_non_cake_queries_unfiltered() -> None:
    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                {"id": "flower001", "name": "Rose Bouquet", "category": {"name": "Flowers"}},
                {"id": "flower002", "name": "Lily Stand Display", "category": {"name": "Flowers"}},
            ],
            "applied_filters": {"q": "flowers", "limit": 10},
        },
    }
    products = extract_search_products(tool_results)
    assert len(products) == 2


def test_build_products_carousel_html_renders_carousel() -> None:
    html = build_products_carousel_html(_SEARCH_TOOL_RESULTS)
    assert html is not None
    assert 'data-testid="product-carousel"' in html
    assert "Chocolate Birthday Cake" in html


def test_build_products_carousel_html_empty_when_no_results() -> None:
    assert build_products_carousel_html({SEARCH_PRODUCTS_TOOL: {"results": []}}) is None


def test_build_products_carousel_html_budget_sorts_in_budget_first() -> None:
    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                _product("expensive", "Premium Hamper", amount=12000.0),
                _product("cheap", "Chocolate Box", amount=4500.0),
                _product("hidden", "Luxury Combo", amount=20000.0),
            ],
            "applied_filters": {"q": "chocolate", "limit": 10},
        },
    }
    html = build_products_carousel_html(tool_results, budget_max=8000.0, currency="LKR")
    assert html is not None
    cheap_idx = html.index('data-product-id="cheap"')
    expensive_idx = html.index('data-product-id="expensive"')
    hidden_idx = html.find('data-product-id="hidden"')
    assert cheap_idx < expensive_idx
    assert hidden_idx == -1


def test_build_products_carousel_html_fallback_uses_budget_refined_last_search() -> None:
    last_search = [
        _product("cheap", "Chocolate Box", amount=4500.0),
        _product("expensive", "Premium Hamper", amount=12000.0),
    ]
    html = build_products_carousel_html(
        None,
        budget_max=6000.0,
        currency="LKR",
        last_search_products=last_search,
    )
    assert html is not None
    assert 'data-product-id="cheap"' in html
    assert 'data-product-id="expensive"' not in html


def test_build_products_carousel_html_stale_fallback_uses_mcp_summary() -> None:
    last_search = [
        {
            "id": "cake1",
            "name": "Springtime Birthday Ribbon Cake",
            "summary": "Delicate sponge with ribbon decoration.",
            "price": {"amount": 5770.0, "currency": "LKR"},
            "in_stock": True,
            "url": "https://www.kapruka.com/cake",
        },
    ]
    html = build_products_carousel_html(
        None,
        last_search_products=last_search,
    )
    assert html is not None
    assert "Delicate sponge with ribbon decoration." in html
    assert "thoughtful Kapruka gift for your occasion" not in html


def test_build_products_carousel_html_visible_products_uses_mcp_summary() -> None:
    products = [
        {
            "id": "cake2",
            "name": "Happy Birthday Symphony Ribbon Cake",
            "summary": "Elegant ribbon cake for celebrations.",
            "price": {"amount": 6500.0, "currency": "LKR"},
            "in_stock": True,
            "url": "https://www.kapruka.com/cake2",
        },
    ]
    html = build_products_carousel_html(None, visible_products=products)
    assert html is not None
    assert "Elegant ribbon cake for celebrations." in html


def test_turn_implies_perishable_gift_chocolate_focus() -> None:
    from graphs.nodes.generate_response import _generate_reply_sync, _turn_implies_perishable_gift

    assert _turn_implies_perishable_gift("thanks", session_product_focus="chocolate")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="I'm here for you.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    _generate_reply_sync(
        mock_client,
        model="gemini-test",
        user_prompt="Customer message:\nbreakup\n\ntool_results:\n{}",
        delivery_context_relevant=False,
    )
    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert "Do not mention delivery city" in config.system_instruction


def test_extract_search_products_filters_puja_for_flowers_when_graph_down() -> None:
    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                _product("puja", "Puja Flower Set", amount=3500.0),
                _product("fruit", "Fruit Basket Deluxe", amount=4500.0),
            ],
            "applied_filters": {"q": "flower fruit", "limit": 10},
        },
    }
    products = extract_search_products(
        tool_results,
        budget_max=5000.0,
        currency="LKR",
        user_message="flowers and fruit basket for Kandy, budget 5000 LKR",
        graph_context_available=False,
    )
    assert [item["id"] for item in products] == ["fruit"]


def test_extract_search_products_demotes_puja_when_graph_up() -> None:
    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                _product("puja", "Puja Flower Set", amount=3500.0),
                _product("fruit", "Fruit Basket Deluxe", amount=4500.0),
            ],
            "applied_filters": {"q": "flower fruit", "limit": 10},
        },
    }
    products = extract_search_products(
        tool_results,
        budget_max=5000.0,
        currency="LKR",
        user_message="flowers and fruit basket for Kandy",
        graph_context_available=True,
    )
    assert [item["id"] for item in products] == ["fruit", "puja"]


def test_build_user_prompt_includes_formatted_budget_cap() -> None:
    prompt = _build_user_prompt(
        "cakes under 5000",
        _SEARCH_TOOL_RESULTS,
        budget_max=5000.0,
        currency="LKR",
    )
    assert "Customer budget cap: Rs. 5,000." in prompt


@pytest.mark.asyncio
async def test_generate_response_carousel_respects_session_budget_max() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="Here are gifts within your budget.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                _product("over", "Deluxe Hamper", amount=17000.0),
                _product("in", "Flower Box", amount=6500.0),
                _product("first", "Chocolate Treat", amount=3500.0),
            ],
        },
    }
    state: AgentState = {
        "messages": [
            HumanMessage(content="wife birthday chocolate flowers ~8000 LKR colombo"),
        ],
        "tool_results": tool_results,
        "session_id": "sess-gen-budget",
        "session_budget_max": 8000.0,
        "currency": "LKR",
    }

    result = await generate_response(state, genai_client=mock_client)
    html = _combined_response_html(result)
    first_idx = html.index('data-testid="product-price"')
    first_price_fragment = html[first_idx : first_idx + 120]
    assert "3,500" in first_price_fragment or "3500" in first_price_fragment
    assert 'data-product-id="over"' not in html

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "Customer budget cap: Rs. 8,000." in call_kwargs["contents"]


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
        "budget_max": None,
    }
    prompt = select_response_system_instruction(metadata)
    assert prompt == UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION
    assert "top 2–3 picks" in prompt.lower() or "top 2-3 picks" in prompt.lower()
    assert "no filler empathy" not in prompt.lower()


def test_select_response_system_instruction_situational_tanglish() -> None:
    metadata: IntentMetadata = {
        "is_situational": True,
        "detected_vernacular": "tanglish",
        "requires_delivery_validation": False,
        "target_city": None,
        "budget_max": None,
        "vernacular_score_hint": 0.45,
    }
    prompt = select_response_system_instruction(metadata)
    assert prompt.startswith(LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION)
    assert "Tanglish" in prompt


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
            "budget_max": None,
        },
        "session_id": "sess-gen-utility",
    }

    await generate_response(state, genai_client=mock_client)

    config = mock_client.models.generate_content.call_args.kwargs["config"]
    instruction = config.system_instruction.lower()
    assert "curate" in instruction or "top 2" in instruction
    assert "no filler empathy" not in instruction


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
            "budget_max": None,
            "vernacular_score_hint": 0.45,
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

    assert 'data-testid="product-carousel"' not in _combined_response_html(result)
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


def test_build_discovery_template_reply_warm_top_three_opener() -> None:
    products = [
        _product("cake00ka002034", "Chocolate Birthday Cake", amount=4500.0),
        _product("cake00ka002099", "Vanilla Celebration Cake", amount=3800.0),
        _product("cake00ka002100", "Strawberry Delight Cake", amount=4200.0),
    ]
    reply = _build_discovery_template_reply(products)
    assert reply.startswith("Here are a few thoughtful Kapruka picks:")
    assert "Chocolate Birthday Cake" in reply
    assert "Vanilla Celebration Cake" in reply
    assert "Strawberry Delight Cake" in reply
    assert "4500.0" not in reply
    assert "LKR 4" not in reply
    assert "Rs. 4,500" in reply


def test_format_product_line_uses_format_currency() -> None:
    line = _format_product_line(
        _product("cake00ka002034", "Chocolate Birthday Cake", amount=8000.0),
    )
    assert "Rs. 8,000" in line
    assert "8000.0" not in line
    assert "LKR 8000" not in line


def test_format_product_line_missing_in_stock_does_not_claim_out_of_stock() -> None:
    product = _product("cake00ka002078", "Pastel Love", amount=5200.0)
    product.pop("in_stock")
    product.pop("stock_level")
    line = _format_product_line(product)
    assert "out of stock" not in line.lower()
    assert "stock not verified" in line.lower()


def test_stock_consistency_guard_rewrites_unavailability_with_in_stock_carousel() -> None:
    products = [
        _product("cake00ka002078", "Pastel Love", amount=5200.0),
    ]
    reply = "Pastel Love is currently out of stock on Kapruka."
    guarded = stock_consistency_guard(reply, products, user_message="Pastel Love cake")
    assert "out of stock" not in guarded.lower()
    assert "Pastel Love" in guarded
    assert "thoughtful Kapruka picks" in guarded


def test_build_discovery_template_reply_prepends_artificial_floral_note() -> None:
    products = [
        _product("EF_PC_CHOC0V571POD00108", "Kit Kat Silk Roses Bouquet", amount=5900.0),
        _product("flower00ka001", "6 Red Rose Bouquet", amount=5210.0),
    ]
    reply = _build_discovery_template_reply(
        products,
        user_message="chocolate and flowers wife birthday",
    )
    assert "artificial" in reply.lower()
    assert "not fresh-cut" in reply.lower()
    assert "Kit Kat Silk Roses Bouquet" in reply


@pytest.mark.asyncio
async def test_generate_response_study_turn_4_silk_roses_disclaimer() -> None:
    """Study turn 4: Kit Kat Silk Roses recommendation includes proactive artificial note."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Here are a few picks for your wife's birthday: Kit Kat Silk Roses Bouquet.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "chocolate flowers birthday"},
            "result": {
                "results": [
                    _product(
                        "EF_PC_CHOC0V571POD00108",
                        "Kit Kat Silk Roses Bouquet",
                        amount=5900.0,
                    ),
                ],
            },
        },
    ]
    state: AgentState = {
        "messages": [HumanMessage(content="chocolate and flowers wife birthday")],
        "intent": "discovery",
        "tool_trace": tool_trace,
        "session_id": "sess-study-turn-4-silk",
    }

    result = await generate_response(state, genai_client=mock_client)

    lower = result["assistant_message"].lower()
    assert "kit kat silk roses bouquet" in lower
    assert "artificial" in lower
    assert "not fresh-cut" in lower


def test_cap_search_products_for_llm_context_limits_to_five() -> None:
    many_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [_product(f"cake{i:03d}", f"Cake {i}", amount=1000.0 + i) for i in range(8)],
        },
    }
    capped = _cap_search_products_for_llm_context(many_results, limit=5)
    assert capped is not None
    results = capped[SEARCH_PRODUCTS_TOOL]["results"]
    assert len(results) == 5
    assert results[0]["id"] == "cake000"
    assert results[4]["id"] == "cake004"
    assert len(many_results[SEARCH_PRODUCTS_TOOL]["results"]) == 8


@pytest.mark.asyncio
async def test_generate_response_caps_products_in_gemini_context() -> None:
    """Gemini prompt receives at most five search products; carousel keeps full list."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="Here are curated cakes from Kapruka.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    many_products = [_product(f"cake{i:03d}", f"Cake {i}", amount=1000.0 + i) for i in range(8)]
    tool_results = {SEARCH_PRODUCTS_TOOL: {"results": many_products}}

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cakes")],
        "tool_results": tool_results,
        "session_id": "sess-gen-cap-llm",
    }

    result = await generate_response(state, genai_client=mock_client)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    context = call_kwargs["contents"]
    assert "Cake 5" not in context
    assert "Cake 4" in context
    assert 'data-product-id="cake007"' in _combined_response_html(result)


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

    html = _combined_response_html(result)
    assert 'data-testid="product-carousel"' in html
    assert 'data-product-id="cake00ka002034"' in html
    assert 'data-product-id="cake00ka002099"' in html
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "Chocolate Birthday Cake" in call_kwargs["contents"]
    assert "Vanilla Celebration Cake" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_generate_response_clarifying_question_with_carousel() -> None:
    """ask_user with fresh search renders clarifier alongside carousel (clarify+search)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="Here are flower options while we confirm delivery.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

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

    assert "Which city should we deliver to?" in result["assistant_message"]
    assert "Which city should we deliver to?" in result["response_html"]
    assert 'data-testid="product-carousel"' in _combined_response_html(result)
    mock_client.models.generate_content.assert_called_once()


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

    html = _combined_response_html(result)
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
    assert "broader gift type" in result["assistant_message"].lower()
    assert 'data-testid="product-carousel"' not in _combined_response_html(result)
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_empty_search_after_broaden_suggests_next_steps() -> None:
    """After broaden retry, empty-state copy references broader options."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom in Kandy under $30")],
        "intent": "discovery",
        "search_broaden_applied": True,
        "tool_trace": [
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "birthday cake", "max_price": 30.0},
                "result": {"results": []},
            },
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "cake", "max_price": 30.0},
                "result": {"results": []},
            },
        ],
        "agent_loop_done": True,
        "session_id": "sess-gen-broaden-empty",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "broadened the search" in result["assistant_message"].lower()
    assert "higher budget" in result["assistant_message"].lower()
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


def test_build_agent_tool_error_message_rate_limit() -> None:
    message = build_agent_tool_error_message(
        tool=SEARCH_PRODUCTS_TOOL,
        raw_message="Rate limit exceeded",
        error_code="429",
    )
    assert "checking our catalog" in message.lower()


def test_build_agent_tool_error_message_city_not_deliverable() -> None:
    message = build_agent_tool_error_message(
        tool=CHECK_DELIVERY_TOOL,
        raw_message="City is not in the Kapruka delivery network",
        error_code="city_not_deliverable",
    )
    assert "cannot deliver to that city" in message.lower()
    assert "Colombo 03" in message
    assert "loc:" not in message.lower()


def test_build_agent_tool_error_message_get_product_unresolved() -> None:
    message = build_agent_tool_error_message(
        tool=GET_PRODUCT_TOOL,
        raw_message="product_id_unresolved",
        error_code="product_id_unresolved",
    )
    assert "carousel" in message.lower()


@pytest.mark.asyncio
async def test_generate_response_product_detail_follow_up_uses_session_weight() -> None:
    """Follow-up detail turns reuse persisted MCP attributes when tool_results reset."""
    mock_client = MagicMock()
    carousel_product = {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Fresh sponge with ribbon.",
        "price": {"amount": 5770.0, "currency": "LKR"},
    }
    state: AgentState = {
        "messages": [HumanMessage(content="what is it")],
        "intent": "discovery",
        "tool_results": {},
        "tool_trace": [],
        "last_search_products": [carousel_product],
        "last_visible_products": [carousel_product],
        "session_resolved_product": {
            "id": "CAKE00KA001685",
            "name": "Springtime Birthday Ribbon Cake",
            "price": {"amount": 5770.0, "currency": "LKR"},
            "attributes": {"weight": "2.77"},
        },
        "session_id": "sess-gen-product-detail-followup",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "2.77 Lbs" in result["assistant_message"]
    assert "CAKE00KA001685" in result["assistant_message"]
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_persists_get_product_in_session() -> None:
    """Fresh kapruka_get_product results are saved for later turns."""
    mock_client = MagicMock()
    get_product_payload = {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Fresh sponge with ribbon.",
        "price": {"amount": 5770.0, "currency": "LKR"},
        "attributes": {"weight": "2.77"},
    }
    state: AgentState = {
        "messages": [HumanMessage(content="Tell me more about the Springtime cake")],
        "intent": "discovery",
        "tool_results": {GET_PRODUCT_TOOL: get_product_payload},
        "tool_trace": [
            {
                "name": GET_PRODUCT_TOOL,
                "args": {"product_id": "CAKE00KA001685"},
                "result": get_product_payload,
            },
        ],
        "last_search_products": [get_product_payload],
        "session_id": "sess-gen-product-detail-persist",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert result.get("session_resolved_product") == {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Fresh sponge with ribbon.",
        "price": {"amount": 5770.0, "currency": "LKR"},
        "attributes": {"weight": "2.77"},
    }
    mock_client.models.generate_content.assert_not_called()


def test_build_agent_tool_error_message_validation_error_hides_pydantic_loc() -> None:
    message = build_agent_tool_error_message(
        tool=CHECK_DELIVERY_TOOL,
        raw_message="delivery_date: Input should be a valid date",
        error_code="validation_error",
    )
    assert "loc:" not in message.lower()
    assert "delivery" in message.lower()


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
    assert 'data-testid="product-carousel"' not in _combined_response_html(result)
    mock_client.models.generate_content.assert_not_called()


def test_delivery_claim_guard_blocks_ungrounded_delivery_fee() -> None:
    """Guard replaces fee/availability claims when kapruka_check_delivery is absent."""
    reply = "Delivery to Colombo is available with a flat delivery rate of Rs. 350 per order."
    guarded = delivery_claim_guard(reply, tool_trace=[])
    assert guarded != reply
    assert "When would you like delivery?" in guarded
    assert "Rs. 350" not in guarded


def test_delivery_claim_guard_allows_grounded_delivery_claim() -> None:
    """Grounded replies pass through when check_delivery ran this turn."""
    reply = "Flat delivery rate: Rs. 350 per order to Colombo."
    trace: list[ToolInvocation] = [
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Colombo 03", "delivery_date": "2026-06-14"},
            "result": {
                "city": "Colombo 03",
                "now": "2026-06-07T10:00:00+05:30",
                "checked_date": "2026-06-14",
                "available": True,
                "rate": 350.0,
                "currency": "LKR",
                "reason": None,
                "next_available_date": None,
                "perishable_warning": None,
            },
        },
    ]
    assert delivery_claim_guard(reply, tool_trace=trace) == reply


def test_delivery_claim_guard_city_date_asks_before_fee() -> None:
    """City + date in user message without MCP check prompts date confirmation, not a fee."""
    reply = "We can deliver to Colombo next Saturday for Rs. 400."
    user_message = "can you deliver flowers to Colombo next Saturday?"
    guarded = delivery_claim_guard(reply, tool_trace=[], user_message=user_message)
    assert "Colombo" in guarded
    assert "won't quote a fee" in guarded
    assert "Rs. 400" not in guarded


def test_delivery_claim_guard_defers_date_for_city_gift_discovery() -> None:
    reply = "I can deliver to Colombo — the delivery fee is Rs. 300."
    state = {
        "intent_metadata": {"target_city": "Colombo"},
    }
    guarded = delivery_claim_guard(
        reply,
        tool_trace=[],
        user_message="I need a birthday cake for my mom in Colombo",
        state=state,
    )
    assert "When would you like" not in guarded
    assert "checkout" in guarded.lower()
    assert "Colombo" in guarded


def test_delivery_claim_guard_skips_discovery_only_turns() -> None:
    reply = "I have not verified Kapruka delivery for that location and date yet."
    guarded = delivery_claim_guard(
        reply,
        tool_trace=[],
        user_message="anniversary gifts",
        delivery_context_relevant=False,
    )
    assert guarded == reply


def test_rate_limit_banner_html_embedded_in_tool_error_response() -> None:
    from graphs.nodes.generate_response import _rate_limit_banner_html

    banner = _rate_limit_banner_html(
        {
            "tool": SEARCH_PRODUCTS_TOOL,
            "error": "rate_limit_exceeded",
            "message": "Rate limit exceeded",
            "retry_after_seconds": "30",
        },
    )
    assert banner is not None
    assert 'data-testid="rate-limit-banner"' in banner
    html = render_assistant_html(
        build_agent_tool_error_message(
            tool=SEARCH_PRODUCTS_TOOL,
            raw_message="Rate limit exceeded",
            error_code="rate_limit_exceeded",
        ),
        rate_limit_banner_html=banner,
    )
    assert "Rate limit exceeded" not in html
    assert "checking our catalog" in html.lower()


def test_carousel_consistency_guard_rewrites_negated_reply_with_products() -> None:
    """Eval B-03: negation copy is replaced when search_products returned carousel picks."""
    products = [
        _product("flower001", "6 Red Rose Bouquet", amount=4500.0),
        _product("flower002", "Blush Roses Combo", amount=4800.0),
    ]
    reply = (
        "I couldn't find any fresh roses within your budget on Kapruka. "
        "You might try widening your search."
    )
    guarded = carousel_consistency_guard(
        reply,
        products,
        user_message="fresh roses under 5000 LKR",
    )
    assert "couldn't find" not in guarded.lower()
    assert "thoughtful Kapruka picks" in guarded
    assert "6 Red Rose Bouquet" in guarded
    assert "Blush Roses Combo" in guarded


def test_carousel_consistency_guard_passes_through_positive_reply() -> None:
    products = [_product("flower001", "6 Red Rose Bouquet", amount=4500.0)]
    reply = "Here are some lovely rose options within your budget."
    assert carousel_consistency_guard(reply, products) == reply


def test_carousel_consistency_guard_skips_when_no_products() -> None:
    reply = "I couldn't find any fresh roses within your budget."
    assert carousel_consistency_guard(reply, []) == reply


def test_carousel_strict_budget_anniversary_under_6000() -> None:
    assert _carousel_strict_budget("anniversary gifts under 6000", 6000.0)


def test_carousel_strict_budget_chocolate_under_6000() -> None:
    assert _carousel_strict_budget("chocolate for wife under 6000", 6000.0)


def test_carousel_strict_budget_false_on_topic_pivot() -> None:
    state: AgentState = {
        "intent_metadata": {"topic_pivot": True},
        "session_budget_max": 6000.0,
    }
    assert not _carousel_strict_budget("Nevermind. Cakes.", 6000.0, state=state)


@pytest.mark.asyncio
async def test_generate_response_roses_under_budget_guard_rewrites_llm_negation() -> None:
    """Eval B-03: carousel and reply agree when LLM falsely claims no in-budget roses."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="I couldn't find any fresh roses under Rs. 5,000 on Kapruka right now.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    tool_results = {
        SEARCH_PRODUCTS_TOOL: {
            "results": [
                _product("flower001", "6 Red Rose Bouquet", amount=4500.0),
                _product("flower002", "Premium Rose Arrangement", amount=4900.0),
            ],
        },
    }
    state: AgentState = {
        "messages": [HumanMessage(content="fresh roses under 5000 LKR")],
        "tool_results": tool_results,
        "session_id": "sess-roses-budget",
        "session_budget_max": 5000.0,
        "currency": "LKR",
        "intent": "discovery",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "couldn't find" not in result["assistant_message"].lower()
    assert "no fresh" not in result["assistant_message"].lower()
    assert "thoughtful Kapruka picks" in result["assistant_message"]
    assert "6 Red Rose Bouquet" in result["assistant_message"]
    assert 'data-testid="product-carousel"' in _combined_response_html(result)


def test_build_verified_city_delivery_line_omits_checked_date() -> None:
    line = _build_verified_city_delivery_line(
        city="Kandy",
        rate=500.0,
        currency="LKR",
    )
    assert line == "Delivery to Kandy: Rs. 500 flat rate per order (verified with Kapruka)"
    assert " on " not in line


def test_apply_perishable_delivery_honesty_preflight_city_only_without_date_copy() -> None:
    """Preflight tool_trace renders city fee line without claiming a delivery date."""
    tool_trace: list[ToolInvocation] = [
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Kandy"},
            "result": {
                "city": "Kandy",
                "now": "2026-06-12T12:00:00+05:30",
                "checked_date": "2026-06-12",
                "available": True,
                "rate": 500.0,
                "currency": "LKR",
                "reason": None,
                "next_available_date": None,
                "perishable_warning": None,
            },
        },
    ]
    reply, delivery_html = _apply_perishable_delivery_honesty(
        delivery_date_clarifying_question(),
        tool_trace,
    )
    assert "Delivery to Kandy: Rs. 500 flat rate per order (verified with Kapruka)" in reply
    assert "on 2026-06-12" not in reply
    assert delivery_html is None


@pytest.mark.asyncio
async def test_generate_response_preflight_trace_renders_deliverable_before_date_ask() -> None:
    """Preflight check_delivery in tool_trace surfaces fee without agent_loop check_delivery."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="can you deliver?")],
        "intent": "discovery",
        "session_delivery_city_canonical": "Kandy",
        "agent_clarifying_question": delivery_date_clarifying_question(),
        "agent_loop_exit_reason": "ask_user",
        "tool_trace": [
            {
                "name": CHECK_DELIVERY_TOOL,
                "args": {"city": "Kandy"},
                "result": {
                    "city": "Kandy",
                    "now": "2026-06-12T12:00:00+05:30",
                    "checked_date": "2026-06-12",
                    "available": True,
                    "rate": 500.0,
                    "currency": "LKR",
                    "reason": None,
                    "next_available_date": None,
                    "perishable_warning": None,
                },
            },
        ],
        "session_id": "sess-preflight-reply",
    }

    result = await generate_response(state, genai_client=mock_client)

    message = result["assistant_message"]
    assert "Delivery to Kandy: Rs. 500 flat rate per order (verified with Kapruka)" in message
    assert delivery_date_clarifying_question() in message
    assert "on 2026-06-12" not in message
    mock_client.models.generate_content.assert_not_called()


def test_build_verified_delivery_fee_line_uses_format_currency() -> None:
    line = _build_verified_delivery_fee_line(
        city="Galle",
        checked_date="2026-06-17",
        rate=450.0,
        currency="LKR",
    )
    assert line == "Delivery to Galle on Wednesday, 17 June 2026: Rs. 450 (verified with Kapruka)"


def test_apply_perishable_delivery_honesty_appends_verified_fee_from_tool_trace() -> None:
    """Grounded check_delivery with rate appends verified fee using canonical city from args."""
    tool_trace: list[ToolInvocation] = [
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Galle", "delivery_date": "2026-06-17"},
            "result": {
                "city": "Galle",
                "now": "2026-06-16T10:00:00+05:30",
                "checked_date": "2026-06-17",
                "available": True,
                "rate": 450.0,
                "currency": "LKR",
                "reason": None,
                "next_available_date": None,
                "perishable_warning": None,
            },
        },
    ]
    reply, delivery_html = _apply_perishable_delivery_honesty(
        "Here are a few roses we can send.",
        tool_trace,
    )
    assert "Delivery to Galle on Wednesday, 17 June 2026: Rs. 450 (verified with Kapruka)" in reply
    assert delivery_html is not None
    assert 'data-testid="delivery-date-available"' in delivery_html
    assert "Flat delivery rate: Rs. 450 per order." in delivery_html


@pytest.mark.asyncio
async def test_generate_response_surfaces_delivery_fee_when_mcp_returns_rate() -> None:
    """Discovery reply quotes verified fee when check_delivery succeeds without LLM fee copy."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Here are some lovely rose bouquets for Galle.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "roses"},
            "result": {
                "results": [_product("flower00ka002", "Classic Roses")],
            },
        },
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Galle", "delivery_date": "2026-06-17"},
            "result": {
                "city": "Galle",
                "now": "2026-06-16T10:00:00+05:30",
                "checked_date": "2026-06-17",
                "available": True,
                "rate": 450.0,
                "currency": "LKR",
                "reason": None,
                "next_available_date": None,
                "perishable_warning": None,
            },
        },
    ]
    state: AgentState = {
        "messages": [HumanMessage(content="roses for Galle tomorrow")],
        "intent": "discovery",
        "tool_trace": tool_trace,
        "session_id": "sess-delivery-fee",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert (
        "Yes, we can deliver to Galle on Wednesday, 17 June 2026. Delivery fee is Rs. 450."
        in (result["assistant_message"])
    )
    assert 'data-testid="delivery-date-available"' in result["response_html"]
    assert 'data-slot="delivery-status"' in result["response_html"]


@pytest.mark.asyncio
async def test_generate_response_perishable_warning_surfaces_in_chat() -> None:
    """Study turn 3 follow-up: grounded fee copy plus perishable_warning partial."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message=(
            "Delivery to Colombo is available on 2026-06-22. "
            "The flat delivery rate is Rs. 350 per order."
        ),
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    perishable_warning = (
        "Fresh flowers are best within 1–2 days of delivery. "
        "Your date is 7 days out — consider ordering closer to the event."
    )
    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "flowers"},
            "result": {
                "results": [_product("flower00ka001", "Blush Roses Bouquet")],
            },
        },
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {
                "city": "Colombo 03",
                "delivery_date": "2026-06-22",
                "product_id": "flower00ka001",
            },
            "result": {
                "city": "Colombo 03",
                "now": "2026-06-15T10:00:00+05:30",
                "checked_date": "2026-06-22",
                "available": True,
                "rate": 350.0,
                "currency": "LKR",
                "reason": None,
                "next_available_date": None,
                "perishable_warning": perishable_warning,
            },
        },
    ]
    state: AgentState = {
        "messages": [HumanMessage(content="What is the delivery fee to Colombo on 2026-06-22?")],
        "intent": "discovery",
        "tool_trace": tool_trace,
        "session_id": "sess-perishable-delivery",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "Rs. 350" in result["assistant_message"]
    assert perishable_warning in result["assistant_message"]
    assert 'data-testid="delivery-date-available"' in result["response_html"]
    assert "text-amber-800" in result["response_html"]
    assert 'data-slot="delivery-status"' in result["response_html"]


@pytest.mark.asyncio
async def test_generate_response_guard_blocks_llm_hallucinated_delivery_fee() -> None:
    """LLM delivery fee without check_delivery this turn is replaced with clarifying copy."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(
        message="Yes, we deliver to Kandy for a flat rate of Rs. 500 per order.",
    )
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom in Kandy")],
        "intent": "discovery",
        "tool_trace": [
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "birthday cake"},
                "result": {
                    "results": [_product("cake00ka002034", "Chocolate Birthday Cake")],
                },
            },
        ],
        "session_id": "sess-guard-delivery-fee",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "Rs. 500" not in result["assistant_message"]
    assert "Kandy" in result["assistant_message"]
    assert "checkout" in result["assistant_message"].lower()


def test_breakup_reply_omits_stale_kandy_delivery() -> None:
    """Situational breakup turns must not append stale Kandy delivery fee lines."""
    tool_trace: list[ToolInvocation] = [
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Kandy", "delivery_date": "2026-06-28"},
            "result": {
                "city": "Kandy",
                "now": "2026-06-25T10:00:00+05:30",
                "checked_date": "2026-06-28",
                "available": True,
                "rate": 500.0,
                "currency": "LKR",
                "reason": None,
                "next_available_date": None,
                "perishable_warning": None,
            },
        },
    ]
    reply, delivery_html = _apply_perishable_delivery_honesty(
        "I'm really sorry you're going through this breakup.",
        tool_trace,
        user_message="We just broke up and I'm heartbroken.",
        delivery_context_relevant=False,
    )
    assert "Kandy" not in reply
    assert "verified with Kapruka" not in reply
    assert delivery_html is None


def test_prepend_situational_empathy_skips_when_reply_already_has_sorry() -> None:
    from graphs.nodes.generate_response import _prepend_situational_empathy

    metadata = {"is_situational": True}
    reply = "I'm really sorry you're going through this breakup. Here are some ideas."
    result = _prepend_situational_empathy(reply, metadata)
    assert result.count("sorry") == 1
    assert result.startswith("I'm really sorry")


@pytest.mark.asyncio
async def test_generate_response_budget_turn_prefers_refined_chocolate_carousel() -> None:
    """Budget-only turn uses last_search chocolate picks, not greeting-card MCP drift."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = AssistantReply(message="Here are chocolate gifts within your budget.")
    mock_response.text = mock_response.parsed.model_dump_json()
    mock_client.models.generate_content.return_value = mock_response

    chocolate_product = {
        "id": "choc001",
        "name": "Heart Chocolate Box",
        "price": {"amount": 4500.0, "currency": "LKR"},
        "in_stock": True,
        "category": {"name": "Chocolate"},
    }
    greeting_card = {
        "id": "card001",
        "name": "Greeting Card",
        "price": {"amount": 1200.0, "currency": "LKR"},
        "in_stock": True,
        "category": {"name": "Greeting Cards"},
    }
    state: AgentState = {
        "intent": "discovery",
        "messages": [HumanMessage(content="under 6000")],
        "session_id": "sess-budget-refine",
        "session_budget_max": 6000.0,
        "session_product_focus": "chocolate",
        "session_search_query": "chocolate gift",
        "last_search_products": [chocolate_product],
        "tool_results": {
            SEARCH_PRODUCTS_TOOL: {
                "results": [greeting_card],
                "applied_filters": {"q": "gift under 6000"},
            },
        },
        "currency": "LKR",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "Heart Chocolate Box" in _combined_response_html(result)
    assert result.get("response_html") is not None
    assert "Greeting Card" not in _combined_response_html(result)


@pytest.mark.asyncio
async def test_generate_response_discovery_search_rate_limit_uses_friendly_copy() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="chocolate gift for my wife, budget 5000")],
        "intent": "discovery",
        "tool_results": {
            SEARCH_PRODUCTS_TOOL: {
                "error": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Wait a moment before retrying.",
                "retry_after_seconds": 60,
            },
        },
        "last_search_products": [
            {
                "id": "CHOC1",
                "name": "Sweet Indulgence Chocolate Gift Box",
                "price": {"amount": 3230.0, "currency": "LKR"},
                "in_stock": True,
            },
        ],
        "session_id": "sess-discovery-rate-limit",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "checking our catalog" in result["assistant_message"].lower()
    assert "Rate limit exceeded" not in result["assistant_message"]


@pytest.mark.asyncio
async def test_generate_response_delivery_fee_skips_product_detail_early_return() -> None:
    mock_client = MagicMock()
    springtime = {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Pastel ribbon cake.",
        "price": {"amount": 5770.0, "currency": "LKR"},
        "attributes": {"weight": "2.77"},
    }
    state: AgentState = {
        "messages": [
            HumanMessage(
                content="Can you deliver to Colombo 05 this Sunday? What's the delivery fee?",
            ),
        ],
        "intent": "discovery",
        "intent_metadata": {
            "target_city": "Colombo 05",
            "delivery_date": "2026-06-28",
            "requires_delivery_validation": True,
        },
        "session_resolved_product": springtime,
        "last_search_products": [springtime],
        "tool_results": {
            CHECK_DELIVERY_TOOL: {
                "city": "Colombo 05",
                "now": "2026-06-28T10:00:00+05:30",
                "checked_date": "2026-06-28",
                "available": True,
                "rate": 300,
                "currency": "LKR",
            },
        },
        "tool_trace": [
            {
                "name": CHECK_DELIVERY_TOOL,
                "args": {"city": "Colombo 05", "delivery_date": "2026-06-28"},
                "result": {
                    "city": "Colombo 05",
                    "now": "2026-06-28T10:00:00+05:30",
                    "checked_date": "2026-06-28",
                    "available": True,
                    "rate": 300,
                    "currency": "LKR",
                },
            },
        ],
        "session_id": "sess-delivery-only-detail-guard",
    }

    result = await generate_response(state, genai_client=mock_client)

    assert "300" in result["assistant_message"]
    assert "Springtime Birthday Ribbon Cake Is A" not in result["assistant_message"]
    assert "product-card" not in (result.get("response_html") or "")
