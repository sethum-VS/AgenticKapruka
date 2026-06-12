"""Synthesize assistant reply from MCP tool results and render HTMX partial."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.templating import get_templates, render_product_carousel, render_tracking_status
from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.model_router import select_model
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, ToolInvocation
from lib.chat.delivery_dates import delivery_date_clarifying_question
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.system_prompts import (
    build_general_welcome_message,
    build_response_system_instruction,
)
from lib.checkout.tracking import extract_order_number, tracking_output_from_tool_results
from lib.genai.errors import is_resource_exhausted
from lib.genai.fallback import generate_content_with_fallback
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.utils.text import decode_html_entities
from lib.zep.memory import format_memory_facts_block

logger = logging.getLogger(__name__)

_LLM_CONTEXT_PRODUCT_LIMIT = 5

_CAKE_QUERY_PATTERN = re.compile(r"\bcakes?\b", re.I)
_CAKE_CATEGORY_PATTERN = re.compile(r"\bcake", re.I)
_CAKE_ID_PREFIX = re.compile(r"^cake", re.I)
_ACCESSORY_BLACKLIST = re.compile(
    r"\b(topper|mould|mold|turning\s+table|cake\s+stand|stand)\b",
    re.I,
)

_TOOL_ERROR_ACTION_LABELS: dict[str, str] = {
    SEARCH_PRODUCTS_TOOL: "search the Kapruka catalog",
    GET_PRODUCT_TOOL: "fetch that product",
    LIST_CATEGORIES_TOOL: "browse categories",
    CHECK_DELIVERY_TOOL: "check delivery",
    LIST_CITIES_TOOL: "list delivery cities",
}

_PAST_DELIVERY_ERROR_CODES = frozenset({"past_delivery_date", "validation_error"})

CHECKOUT_REVIEW_SYSTEM_INSTRUCTION = (
    "You are the Kapruka gift shopping assistant at the final checkout review step.\n\n"
    "Synthesize a clear, warm confirmation message using ONLY the checkout summary "
    "JSON provided.\n\n"
    "Rules:\n"
    "- Summarize cart items, delivery, recipient, and sender without inventing facts.\n"
    "- Ask the customer to confirm the order looks correct before payment.\n"
    "- Mention that the next step will provide a secure Kapruka checkout link.\n"
    "- Keep the reply under 150 words.\n"
)


class AssistantReply(BaseModel):
    """Structured Gemini response for the assistant message body."""

    message: str


def merge_tool_trace(tool_trace: list[ToolInvocation]) -> dict[str, Any]:
    """Merge agent-loop invocations into tool_results shape for generate_response.

    Non-search tools use last-wins per tool name. ``kapruka_search_products`` unions
    product dicts across trace entries with deduplication by product id; other search
    payload fields come from the last search invocation.
    """
    merged: dict[str, Any] = {}
    search_products_by_id: dict[str, dict[str, Any]] = {}
    last_search_payload: dict[str, Any] | None = None

    for invocation in tool_trace:
        name = invocation["name"]
        result = invocation["result"]

        if name == SEARCH_PRODUCTS_TOOL and isinstance(result, dict):
            last_search_payload = result
            raw_results = result.get("results")
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict):
                        product_id = item.get("id")
                        if product_id:
                            search_products_by_id[str(product_id)] = item
            continue

        merged[name] = result

    if last_search_payload is not None:
        search_merged = dict(last_search_payload)
        search_merged["results"] = list(search_products_by_id.values())
        merged[SEARCH_PRODUCTS_TOOL] = search_merged

    return merged


def _error_code_from_tool_trace(
    tool_trace: list[ToolInvocation] | None,
    tool_name: str,
) -> str | None:
    """Return MCP error code for the last failed invocation of tool_name."""
    if not tool_trace:
        return None
    for invocation in reversed(tool_trace):
        if invocation.get("name") != tool_name:
            continue
        result = invocation.get("result")
        if isinstance(result, dict) and result.get("error"):
            code = result.get("error")
            return str(code) if code is not None else None
    return None


def build_agent_tool_error_message(
    *,
    tool: str,
    raw_message: str,
    error_code: str | None = None,
) -> str:
    """Tier-1 user-facing copy for agent-loop MCP failures (problem + cause + fix)."""
    if (
        error_code in _PAST_DELIVERY_ERROR_CODES
        and tool == CHECK_DELIVERY_TOOL
        and ("past" in raw_message.lower() or error_code == "past_delivery_date")
    ):
        return delivery_date_clarifying_question()
    if error_code == "date_not_deliverable":
        return (
            "That delivery date is not available. "
            f"{raw_message} "
            "Would you like to try a different date?"
        )

    action = _TOOL_ERROR_ACTION_LABELS.get(tool, "complete that request")
    cause = raw_message.strip() or "Kapruka could not process the request."
    return f"I could not {action} right now. {cause} Please adjust your request and try again."


def _is_general_welcome_path(state: AgentState) -> bool:
    """True when general intent finished with no MCP catalog or tracking payloads."""
    if state.get("intent") != "general":
        return False
    tool_trace = state.get("tool_trace")
    if tool_trace:
        return False
    tool_results = state.get("tool_results")
    return not isinstance(tool_results, dict) or not tool_results


def _resolve_effective_tool_results(state: AgentState) -> dict[str, Any] | None:
    """Prefer checkout/tracking payloads; else merged agent-loop trace; else tool_results."""
    tool_results = state.get("tool_results")
    intent = state.get("intent")
    if isinstance(tool_results, dict) and (
        intent in ("checkout", "tracking") or CHECKOUT_TOOL_KEY in tool_results
    ):
        return tool_results
    tool_trace = state.get("tool_trace")
    if tool_trace:
        return merge_tool_trace(tool_trace)
    return tool_results


def _format_tool_results_context(tool_results: dict[str, Any] | None) -> str:
    """Serialize tool_results for the LLM context block."""
    payload = tool_results or {}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _search_query_from_payload(search_payload: dict[str, Any]) -> str | None:
    filters = search_payload.get("applied_filters")
    if isinstance(filters, dict):
        query = filters.get("q")
        if isinstance(query, str) and query.strip():
            return query.strip()
    return None


def _product_category_text(product: dict[str, Any]) -> str:
    category = product.get("category")
    if isinstance(category, dict):
        parts = [str(category.get(key) or "") for key in ("name", "slug", "id")]
        return " ".join(parts)
    return ""


def _is_cake_search_query(query: str | None) -> bool:
    return bool(query and _CAKE_QUERY_PATTERN.search(query))


def _is_likely_cake_product(product: dict[str, Any]) -> bool:
    product_id = str(product.get("id") or "")
    if _CAKE_ID_PREFIX.match(product_id):
        return True
    name = str(product.get("name") or "")
    if _CAKE_QUERY_PATTERN.search(name):
        return True
    return bool(_CAKE_CATEGORY_PATTERN.search(_product_category_text(product)))


def _is_cake_accessory(product: dict[str, Any]) -> bool:
    name = str(product.get("name") or "")
    summary = str(product.get("summary") or "")
    return bool(_ACCESSORY_BLACKLIST.search(f"{name} {summary}"))


def _filter_cake_search_products(
    products: list[dict[str, Any]],
    query: str | None,
) -> list[dict[str, Any]]:
    """Drop non-cake items and baking accessories when the search q targets cakes."""
    if not _is_cake_search_query(query):
        return products
    return [
        product
        for product in products
        if _is_likely_cake_product(product) and not _is_cake_accessory(product)
    ]


def _curated_search_results(search_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = search_payload.get("results")
    if not isinstance(raw_results, list):
        return []
    products = [
        item
        for item in raw_results
        if isinstance(item, dict) and item.get("id") and item.get("name")
    ]
    query = _search_query_from_payload(search_payload)
    curated = _filter_cake_search_products(products, query)
    if not curated and products and _is_cake_search_query(query):
        return products
    return curated


def _cap_search_products_for_llm_context(
    tool_results: dict[str, Any] | None,
    *,
    limit: int = _LLM_CONTEXT_PRODUCT_LIMIT,
) -> dict[str, Any] | None:
    """Slice curated kapruka_search_products results before Gemini synthesis."""
    if not tool_results:
        return tool_results

    search_payload = tool_results.get(SEARCH_PRODUCTS_TOOL)
    if not isinstance(search_payload, dict):
        return tool_results

    raw_results = search_payload.get("results")
    if not isinstance(raw_results, list):
        return tool_results

    curated = _curated_search_results(search_payload)
    capped_results = curated[:limit]
    if capped_results == raw_results:
        return tool_results

    capped = dict(tool_results)
    capped_search = dict(search_payload)
    capped_search["results"] = capped_results
    capped[SEARCH_PRODUCTS_TOOL] = capped_search
    return capped


def _build_user_prompt(user_message: str, tool_results: dict[str, Any] | None) -> str:
    """Combine user turn and MCP payload for response synthesis."""
    context = _format_tool_results_context(tool_results)
    return (
        f"Customer message:\n{user_message}\n\n"
        f"tool_results (sole source of truth for catalog facts):\n{context}"
    )


def _parse_reply_response(response: types.GenerateContentResponse) -> str:
    """Parse structured or JSON text assistant reply from Gemini."""
    if response.parsed is not None:
        if isinstance(response.parsed, AssistantReply):
            return response.parsed.message.strip()
        validated = AssistantReply.model_validate(response.parsed)
        return validated.message.strip()

    raw_text = (response.text or "").strip()
    if not raw_text:
        msg = "Gemini returned empty assistant reply"
        raise ValueError(msg)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"Gemini reply is not valid JSON: {raw_text!r}"
        raise ValueError(msg) from exc

    try:
        return AssistantReply.model_validate(payload).message.strip()
    except ValidationError as exc:
        msg = f"Gemini reply JSON failed validation: {payload!r}"
        raise ValueError(msg) from exc


def _generate_reply_sync(
    client: genai.Client | None,
    *,
    model: str,
    user_prompt: str,
    zep_memory_facts: list[str] | None = None,
    intent_metadata: IntentMetadata | None = None,
    system_instruction: str | None = None,
    intent: str | None = None,
) -> str:
    """Blocking Gemini call; run via asyncio.to_thread from generate_response."""
    instruction = system_instruction or build_response_system_instruction(
        intent_metadata,
        zep_memory_facts=zep_memory_facts,
        intent=intent,
    )
    if system_instruction is not None and zep_memory_facts:
        instruction += format_memory_facts_block(zep_memory_facts)

    response = generate_content_with_fallback(
        client=client,
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            response_mime_type="application/json",
            response_schema=AssistantReply,
            temperature=0.2,
        ),
    )
    return _parse_reply_response(response)


def extract_search_products(tool_results: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return curated product dicts from kapruka_search_products tool_results, if any."""
    if not tool_results:
        return []

    search_payload = tool_results.get(SEARCH_PRODUCTS_TOOL)
    if not isinstance(search_payload, dict):
        return []

    return _curated_search_results(search_payload)


def build_tracking_status_html(tool_results: dict[str, Any] | None) -> str | None:
    """Render tracking_status partial when kapruka_track_order returned results."""
    tracking = tracking_output_from_tool_results(tool_results)
    if tracking is None:
        return None
    return render_tracking_status(tracking=tracking)


def _build_tracking_assistant_message(tool_results: dict[str, Any] | None) -> str | None:
    """Synthesize a tracking reply from kapruka_track_order tool_results."""
    tracking = tracking_output_from_tool_results(tool_results)
    if tracking is None:
        return None
    return (
        f"Here is the latest status for order {tracking.order_number}: "
        f"{tracking.status_display}. Expected delivery on {tracking.delivery_date}."
    )


def _format_product_line(product: dict[str, Any]) -> str:
    """Single-line catalog summary for template discovery replies."""
    name = decode_html_entities(str(product.get("name") or "item"))
    raw_price = product.get("price")
    price: dict[str, Any] = raw_price if isinstance(raw_price, dict) else {}
    amount = price.get("amount")
    currency = price.get("currency") or "LKR"
    stock_level = product.get("stock_level")
    if isinstance(stock_level, str) and stock_level.strip():
        stock_note = f"in stock ({stock_level.strip().lower()})"
    elif product.get("in_stock"):
        stock_note = "in stock"
    else:
        stock_note = "out of stock"
    if amount is not None:
        return f"'{name}' for {currency} {amount}, {stock_note}"
    return f"'{name}', {stock_note}"


def _build_discovery_template_reply(products: list[dict[str, Any]]) -> str:
    """Deterministic assistant copy from MCP search results (no Gemini)."""
    if not products:
        return ""
    picks = products[:3]
    lines = [_format_product_line(product) for product in picks]
    opener = "Here are a few thoughtful Kapruka picks:"
    if len(lines) == 1:
        return f"{opener} {lines[0]}."
    if len(lines) == 2:
        return f"{opener} {lines[0]}, and {lines[1]}."
    return f"{opener} {lines[0]}, {lines[1]}, and {lines[2]}."


def build_products_carousel_html(tool_results: dict[str, Any] | None) -> str | None:
    """Render product carousel partial when search_products returned results."""
    products = extract_search_products(tool_results)
    if not products:
        return None
    return render_product_carousel(products)


def _build_checkout_assistant_message(tool_results: dict[str, Any] | None) -> str | None:
    """Synthesize a checkout-step reply from run_checkout_graph tool_results."""
    if not tool_results:
        return None
    checkout = tool_results.get(CHECKOUT_TOOL_KEY)
    if not isinstance(checkout, dict):
        return None

    errors = checkout.get("validation_errors")
    if isinstance(errors, dict) and errors:
        first_error = next(iter(errors.values()))
        return str(first_error)

    cart_items = checkout.get("cart_items")
    if not isinstance(cart_items, list) or not cart_items:
        return "Your cart is empty. Add a gift before starting checkout."

    count = sum(int(item.get("quantity", 1)) for item in cart_items if isinstance(item, dict))
    step = checkout.get("current_step") or "cart"
    if step == "cart":
        noun = "item" if count == 1 else "items"
        return (
            f"Let's check out your {count} cart {noun}. "
            "Next, tell me the delivery city for your order."
        )
    if step == "finalize":
        checkout_url = checkout.get("checkout_url")
        order_ref = checkout.get("order_ref")
        if isinstance(checkout_url, str) and checkout_url.strip():
            ref_note = f" (reference {order_ref})" if order_ref else ""
            return (
                f"Your Kapruka order is ready{ref_note}. "
                "Use the button below to pay securely before the link expires."
            )
    return "Continuing your Kapruka checkout."


def render_assistant_html(
    message: str,
    *,
    products_html: str | None = None,
    checkout_review_html: str | None = None,
    checkout_payment_html: str | None = None,
    tracking_status_html: str | None = None,
) -> str:
    """Render templates/chat/message_assistant.html for HTMX swap."""
    templates = get_templates()
    template = templates.env.get_template("chat/message_assistant.html")
    return template.render(
        message=message,
        products_html=products_html,
        checkout_review_html=checkout_review_html,
        checkout_payment_html=checkout_payment_html,
        tracking_status_html=tracking_status_html,
    )


def _extract_checkout_payload(tool_results: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tool_results:
        return None
    checkout = tool_results.get(CHECKOUT_TOOL_KEY)
    if isinstance(checkout, dict):
        return checkout
    return None


def _build_checkout_review_prompt(user_message: str, checkout: dict[str, Any]) -> str:
    summary = {
        key: checkout.get(key)
        for key in (
            "cart_items",
            "delivery_address",
            "delivery_city",
            "delivery_location_type",
            "delivery_date",
            "delivery_instructions",
            "recipient_name",
            "recipient_phone",
            "sender_name",
            "sender_anonymous",
            "gift_message",
        )
    }
    context = json.dumps(summary, indent=2, ensure_ascii=False)
    return (
        f"Customer message:\n{user_message}\n\n"
        f"checkout_summary (sole source of truth for order facts):\n{context}"
    )


async def generate_response(
    state: AgentState,
    *,
    genai_client: genai.Client | None = None,
) -> dict[str, Any]:
    """LangGraph node: synthesize assistant text and render response_html partial."""
    messages = state.get("messages") or []
    user_message = _extract_latest_user_message(messages)
    tool_results = _resolve_effective_tool_results(state)

    if not user_message.strip():
        welcome = build_general_welcome_message()
        return {
            "response_html": render_assistant_html(welcome),
            "assistant_message": welcome,
        }

    if _is_general_welcome_path(state):
        welcome = build_general_welcome_message()
        return {
            "response_html": render_assistant_html(welcome),
            "assistant_message": welcome,
        }

    clarifying_question = state.get("agent_clarifying_question")
    if (
        state.get("agent_loop_exit_reason") == "ask_user"
        and isinstance(clarifying_question, str)
        and clarifying_question.strip()
    ):
        question = clarifying_question.strip()
        return {
            "response_html": render_assistant_html(question),
            "assistant_message": question,
        }

    agent_tool_error = state.get("agent_tool_error")
    if (
        state.get("agent_loop_exit_reason") == "tool_error"
        and isinstance(agent_tool_error, dict)
        and agent_tool_error.get("tool")
        and agent_tool_error.get("message")
    ):
        tool_name = str(agent_tool_error["tool"])
        raw_message = str(agent_tool_error["message"])
        error_code = _error_code_from_tool_trace(state.get("tool_trace"), tool_name)
        error_reply = build_agent_tool_error_message(
            tool=tool_name,
            raw_message=raw_message,
            error_code=error_code,
        )
        return {
            "response_html": render_assistant_html(error_reply),
            "assistant_message": error_reply,
        }

    if state.get("intent") == "tracking":
        tracking_reply = _build_tracking_assistant_message(tool_results)
        if tracking_reply:
            tracking_html = build_tracking_status_html(tool_results)
            return {
                "response_html": render_assistant_html(
                    tracking_reply,
                    tracking_status_html=tracking_html,
                ),
                "assistant_message": tracking_reply,
            }
        if not extract_order_number(user_message):
            missing_number_reply = (
                "Please share your Kapruka order number from your confirmation email. "
                "Use the post-payment number (for example VIMP34456CB2), not the "
                "pre-payment checkout reference that starts with ORD-."
            )
            return {
                "response_html": render_assistant_html(missing_number_reply),
                "assistant_message": missing_number_reply,
            }

    if state.get("intent") == "checkout":
        checkout = _extract_checkout_payload(tool_results)
        if state.get("checkout_state") == "review" and checkout:
            review_html = checkout.get("review_html")
            review_html_str = (
                review_html if isinstance(review_html, str) and review_html.strip() else None
            )

            client = genai_client
            model = select_model(state)
            user_prompt = _build_checkout_review_prompt(user_message, checkout)
            zep_memory_facts = state.get("zep_memory_facts")
            reply_text = await asyncio.to_thread(
                _generate_reply_sync,
                client,
                model=model,
                user_prompt=user_prompt,
                zep_memory_facts=zep_memory_facts,
                system_instruction=CHECKOUT_REVIEW_SYSTEM_INSTRUCTION,
            )
            if not reply_text:
                reply_text = (
                    "Please review your order summary below and "
                    "confirm when everything looks correct."
                )

            return {
                "response_html": render_assistant_html(
                    reply_text,
                    checkout_review_html=review_html_str,
                ),
                "assistant_message": reply_text,
                "model_tier": "pro",
            }

        checkout_reply = _build_checkout_assistant_message(tool_results)
        if checkout_reply:
            payment_html = checkout.get("payment_cta_html") if checkout else None
            payment_html_str = (
                payment_html if isinstance(payment_html, str) and payment_html.strip() else None
            )
            return {
                "response_html": render_assistant_html(
                    checkout_reply,
                    checkout_payment_html=payment_html_str,
                ),
                "assistant_message": checkout_reply,
            }

    if state.get("intent") == "discovery":
        search_payload = (tool_results or {}).get(SEARCH_PRODUCTS_TOOL)
        if isinstance(search_payload, dict):
            error_message = search_payload.get("message")
            if search_payload.get("error") and isinstance(error_message, str):
                return {
                    "response_html": render_assistant_html(error_message),
                    "assistant_message": error_message,
                }
            if search_payload.get("results") == []:
                empty_reply = (
                    "I couldn't find products matching that search on Kapruka. "
                    "Try naming a specific gift type, such as birthday cake, flowers, "
                    "or chocolates."
                )
                return {
                    "response_html": render_assistant_html(empty_reply),
                    "assistant_message": empty_reply,
                }

    products = extract_search_products(tool_results)
    products_html = build_products_carousel_html(tool_results)

    client = genai_client
    model = select_model(state)
    user_prompt = _build_user_prompt(
        user_message,
        _cap_search_products_for_llm_context(tool_results),
    )

    zep_memory_facts = state.get("zep_memory_facts")
    intent_metadata = state.get("intent_metadata")
    intent = state.get("intent")
    try:
        reply_text = await asyncio.to_thread(
            _generate_reply_sync,
            client,
            model=model,
            user_prompt=user_prompt,
            zep_memory_facts=zep_memory_facts,
            intent_metadata=intent_metadata,
            intent=intent,
        )
    except Exception as exc:
        if not is_resource_exhausted(exc):
            raise
        reply_text = _build_discovery_template_reply(products)
        logger.warning(
            "generate_response: Gemini rate limited; template fallback (%d products)",
            len(products),
            exc_info=True,
        )

    if not reply_text:
        reply_text = _build_discovery_template_reply(products) or (
            "I could not generate a response. Please try again."
        )

    reply_text = decode_html_entities(reply_text)

    logger.info(
        "generate_response: rendered assistant reply (%d chars, carousel=%s)",
        len(reply_text),
        bool(products_html),
    )
    return {
        "response_html": render_assistant_html(reply_text, products_html=products_html),
        "assistant_message": reply_text,
    }
