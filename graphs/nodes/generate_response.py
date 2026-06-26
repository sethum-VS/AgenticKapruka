"""Synthesize assistant reply from MCP tool results and render HTMX partial."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from typing import Any, cast

from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, ValidationError

from app.templating import (
    get_templates,
    render_cart_partial_oob,
    render_delivery_date_status,
    render_product_carousel,
    render_rate_limit_banner,
    render_tracking_status,
)
from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.model_router import select_model
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, ToolInvocation
from lib.chat.delivery_dates import delivery_date_clarifying_question, normalize_delivery_date
from lib.chat.intent_heuristics import is_budget_refinement_message
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.off_topic import impossible_request_subject, off_topic_topic
from lib.chat.product_curation import (
    carousel_focus_guard,
    curate_carousel_products,
    filter_cake_accessories,
    has_graph_hybrid_context,
    is_cake_accessory,
    refine_last_search_by_budget,
)
from lib.chat.product_detail import (
    is_delivery_fee_question,
    is_product_detail_turn,
    match_product_from_last_search,
    summarize_product_from_carousel,
)
from lib.chat.product_honesty import (
    artificial_floral_note_for_picks,
    reply_already_discloses_artificial_floral,
)
from lib.chat.query_preprocessor import extract_target_city, is_delivery_context_relevant_turn
from lib.chat.search_broadening import build_empty_search_reply
from lib.chat.status_copy import PUTTING_TOGETHER_RECOMMENDATIONS
from lib.chat.system_prompts import (
    build_farewell_message,
    build_general_welcome_message,
    build_impossible_product_redirect,
    build_off_topic_redirect_message,
    build_response_system_instruction,
    is_farewell_message,
)
from lib.checkout.tracking import (
    OrderReferenceKind,
    build_missing_tracking_number_message,
    build_tracking_failure_message,
    classify_order_reference,
    extract_order_number,
    tracking_error_from_tool_results,
    tracking_output_from_tool_results,
)
from lib.genai.errors import is_resource_exhausted
from lib.genai.fallback import generate_content_with_fallback
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import CheckDeliveryOutput
from lib.neo4j.hybrid_context import extract_budget
from lib.redis.cart import StoredCartItem
from lib.utils.currency import format_currency
from lib.utils.text import decode_html_entities, normalize_catalog_text
from lib.utils.timezone import colombo_today, format_delivery_date_friendly
from lib.zep.memory import format_memory_facts_block, scope_memory_facts_for_turn

logger = logging.getLogger(__name__)

_LLM_CONTEXT_PRODUCT_LIMIT = 5

_CAKE_QUERY_PATTERN = re.compile(r"\bcakes?\b", re.I)
_CAKE_CATEGORY_PATTERN = re.compile(r"\bcake", re.I)
_CAKE_ID_PREFIX = re.compile(r"^cake", re.I)

_TOOL_ERROR_ACTION_LABELS: dict[str, str] = {
    SEARCH_PRODUCTS_TOOL: "search the Kapruka catalog",
    GET_PRODUCT_TOOL: "fetch that product",
    LIST_CATEGORIES_TOOL: "browse categories",
    CHECK_DELIVERY_TOOL: "check delivery",
    LIST_CITIES_TOOL: "list delivery cities",
    TRACK_ORDER_TOOL: "look up that order",
}

_PAST_DELIVERY_ERROR_CODES = frozenset({"past_delivery_date", "validation_error"})

_DELIVERY_FEE_CLAIM = re.compile(
    r"\b(?:delivery\s+fee|flat\s+(?:delivery\s+)?rate|delivery\s+(?:rate|charge|cost))\b",
    re.I,
)
_DELIVERY_AVAILABILITY_CLAIM = re.compile(
    r"\b(?:delivery\s+available|can\s+deliver|we\s+deliver|deliver(?:s|ed)?\s+to|"
    r"not\s+available\s+for\s+delivery|unable\s+to\s+deliver)\b",
    re.I,
)
_DELIVERY_QUOTED_RATE = re.compile(
    r"\b(?:rs\.?|lkr)\s*[\d,]+(?:\.\d+)?\s*(?:per\s+order|for\s+delivery|delivery)\b",
    re.I,
)
_CAROUSEL_NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"couldn'?t\s+find|could\s+not\s+find|"
    r"no\s+fresh|"
    r"none\s+within|"
    r"no\s+options\s+under|"
    r"no\s+products|nothing\s+matching|don'?t\s+have\s+any"
    r")\b",
    re.I,
)

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


def _turn_has_fresh_search(tool_trace: list[ToolInvocation] | None) -> bool:
    return any(invocation.get("name") == SEARCH_PRODUCTS_TOOL for invocation in (tool_trace or []))


def _turn_search_has_products(tool_trace: list[ToolInvocation] | None) -> bool:
    """True when the latest search_products invocation returned at least one product."""
    if not tool_trace:
        return False
    for invocation in reversed(tool_trace):
        if invocation.get("name") != SEARCH_PRODUCTS_TOOL:
            continue
        result = invocation.get("result")
        if not isinstance(result, dict) or result.get("error"):
            return False
        raw_results = result.get("results")
        if not isinstance(raw_results, list):
            return False
        return any(isinstance(item, dict) for item in raw_results)
    return False


def _session_budget_applies(state: AgentState, user_message: str) -> bool:
    pivot_meta = state.get("intent_metadata") or {}
    if isinstance(pivot_meta, dict) and pivot_meta.get("topic_pivot"):
        return False
    if extract_budget(user_message) is not None:
        return True
    if is_budget_refinement_message(user_message):
        return True
    messages = state.get("messages") or []
    user_turns = [message for message in messages if isinstance(message, HumanMessage)]
    if len(user_turns) >= 2:
        prior = user_turns[-2].content
        if isinstance(prior, str) and extract_budget(prior) is not None:
            return True
    return False


def _suppress_delivery_tool_results(
    tool_results: dict[str, Any] | None,
    *,
    delivery_context_relevant: bool,
) -> dict[str, Any] | None:
    if delivery_context_relevant or not tool_results:
        return tool_results
    if CHECK_DELIVERY_TOOL not in tool_results:
        return tool_results
    filtered = dict(tool_results)
    filtered.pop(CHECK_DELIVERY_TOOL, None)
    return filtered


def _rate_limit_banner_html(agent_tool_error: dict[str, str]) -> str | None:
    error_code = agent_tool_error.get("error")
    if error_code not in ("429", "rate_limit_exceeded"):
        return None
    raw_retry = agent_tool_error.get("retry_after_seconds")
    retry_after = 30
    if isinstance(raw_retry, str) and raw_retry.isdigit():
        retry_after = max(1, int(raw_retry))
    return render_rate_limit_banner(
        title="Still searching…",
        message="I'm checking our catalog — one moment.",
        error_code="rate_limit_exceeded",
        retry_after_seconds=retry_after,
    )


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
    order_number: str | None = None,
    reference_kind: OrderReferenceKind | None = None,
) -> str:
    """Tier-1 user-facing copy for agent-loop MCP failures (problem + cause + fix)."""
    if tool == TRACK_ORDER_TOOL:
        kind: OrderReferenceKind = reference_kind or (
            classify_order_reference(order_number) if order_number else "unknown"
        )
        return build_tracking_failure_message(
            order_number=order_number,
            reference_kind=kind,
            error_code=error_code,
            raw_message=raw_message,
        )
    if (
        error_code in _PAST_DELIVERY_ERROR_CODES
        and tool == CHECK_DELIVERY_TOOL
        and ("past" in raw_message.lower() or error_code == "past_delivery_date")
    ):
        return delivery_date_clarifying_question()
    if error_code in ("429", "rate_limit_exceeded"):
        return "I'm checking our catalog — one moment."
    if error_code == "date_not_deliverable":
        return (
            "That delivery date is not available. "
            f"{raw_message} "
            "Would you like to try a different date?"
        )
    if error_code == "city_not_deliverable" and tool == CHECK_DELIVERY_TOOL:
        return (
            "We cannot deliver to that city. Please choose a Kapruka delivery area "
            "(for example Colombo 03, Kandy, or Galle)."
        )
    if error_code == "product_id_unresolved" and tool == GET_PRODUCT_TOOL:
        return (
            "I couldn't load that product's details — try tapping it in the carousel above, "
            "or tell me which item you mean."
        )
    if error_code == "validation_error":
        lowered = raw_message.lower()
        if tool == GET_PRODUCT_TOOL and (
            error_code == "product_id_unresolved"
            or "product_id_unresolved" in lowered
            or "product_id" in lowered
        ):
            return (
                "I couldn't load that product's details — try tapping it in the carousel above, "
                "or tell me which item you mean."
            )
        if tool == CHECK_DELIVERY_TOOL and (
            "delivery_date" in lowered or "date" in lowered or "past" in lowered
        ):
            return delivery_date_clarifying_question()
        if tool == CHECK_DELIVERY_TOOL and "city" in lowered:
            return (
                "Please choose a valid Kapruka delivery city "
                "(for example Colombo 03, Kandy, or Galle)."
            )
        return "Please check your delivery details and try again."

    action = _TOOL_ERROR_ACTION_LABELS.get(tool, "complete that request")
    cause = raw_message.strip() or "Kapruka could not process the request."
    return f"I could not {action} right now. {cause} Please adjust your request and try again."


def _reply_claims_delivery_facts(reply_text: str) -> bool:
    """True when assistant copy quotes delivery fees or deliverability without MCP grounding."""
    if not reply_text.strip():
        return False
    if _DELIVERY_FEE_CLAIM.search(reply_text):
        return True
    if _DELIVERY_AVAILABILITY_CLAIM.search(reply_text):
        return True
    if _DELIVERY_QUOTED_RATE.search(reply_text):
        return True
    has_amount = bool(re.search(r"\b[\d,]+(?:\.\d+)?\s*(?:LKR|Rs\.?)\b", reply_text, re.I))
    has_delivery_word = bool(re.search(r"\bdeliver", reply_text, re.I))
    return has_amount and has_delivery_word


def _tool_trace_has_check_delivery(tool_trace: list[ToolInvocation] | None) -> bool:
    return any(invocation.get("name") == CHECK_DELIVERY_TOOL for invocation in (tool_trace or []))


def _user_named_city_and_date(user_message: str) -> bool:
    city = extract_target_city(user_message)
    if not city:
        return False
    return normalize_delivery_date({}, user_message) is not None


def delivery_claim_guard(
    reply_text: str,
    tool_trace: list[ToolInvocation] | None,
    *,
    user_message: str = "",
    delivery_city_status: str | None = None,
    delivery_city_confirmed: bool = False,
    delivery_context_relevant: bool = True,
) -> str:
    """Replace ungrounded delivery fee/availability claims when check_delivery is absent."""
    if not delivery_context_relevant:
        return reply_text
    if not _reply_claims_delivery_facts(reply_text):
        return reply_text
    if _tool_trace_has_check_delivery(tool_trace):
        return reply_text
    if _user_named_city_and_date(user_message):
        city = extract_target_city(user_message) or "your city"
        return (
            f"I can verify Kapruka delivery to {city} once we confirm the date — "
            "I won't quote a fee until that's checked. "
            f"{delivery_date_clarifying_question()}"
        )
    return (
        "I have not verified Kapruka delivery for that location and date yet. "
        f"{delivery_date_clarifying_question()}"
    )


def carousel_consistency_guard(
    reply_text: str,
    products: list[dict[str, Any]],
    *,
    user_message: str = "",
    budget_max: float | None = None,
    currency: str = "LKR",
    strict_budget: bool = False,
) -> str:
    """Replace contradictory empty-search copy when MCP search returned carousel products."""
    if not products or not reply_text.strip():
        return reply_text
    if not _CAROUSEL_NEGATION_PATTERN.search(reply_text):
        return reply_text
    if budget_max is not None and budget_max > 0:
        in_budget = [
            product
            for product in products
            if not product.get("slightly_over_budget") and not product.get("over_budget")
        ]
        if not in_budget and products:
            budget_label = format_currency(budget_max, currency)
            if strict_budget:
                return (
                    f"I couldn't find Kapruka options within your {budget_label} budget. "
                    "Try a slightly higher budget or a broader gift type."
                )
            return (
                f"Here are some Kapruka options; a few exceed your {budget_label} budget — "
                "I've marked those in the carousel."
            )
    return _build_discovery_template_reply(products, user_message=user_message) or reply_text


def _last_check_delivery_invocation(
    tool_trace: list[ToolInvocation] | None,
) -> ToolInvocation | None:
    if not tool_trace:
        return None
    for invocation in reversed(tool_trace):
        if invocation.get("name") != CHECK_DELIVERY_TOOL:
            continue
        result = invocation.get("result")
        if isinstance(result, dict) and not result.get("error"):
            return invocation
    return None


def _canonical_city_from_check_delivery_invocation(invocation: ToolInvocation) -> str | None:
    args = invocation.get("args")
    if isinstance(args, dict):
        raw_city = args.get("city")
        if isinstance(raw_city, str) and raw_city.strip():
            return raw_city.strip()
    result = invocation.get("result")
    if isinstance(result, dict):
        raw_city = result.get("city")
        if isinstance(raw_city, str) and raw_city.strip():
            return raw_city.strip()
    return None


def _is_city_only_check_delivery(invocation: ToolInvocation) -> bool:
    """True when check_delivery ran with city only (preflight before date ask)."""
    args = invocation.get("args")
    if not isinstance(args, dict):
        return False
    delivery_date = args.get("delivery_date")
    return not (isinstance(delivery_date, str) and delivery_date.strip())


def _build_verified_delivery_fee_line(
    *,
    city: str,
    checked_date: str,
    rate: float,
    currency: str,
) -> str:
    fee = format_currency(rate, currency)
    friendly_date = format_delivery_date_friendly(checked_date)
    return f"Delivery to {city} on {friendly_date}: {fee} (verified with Kapruka)"


def _build_verified_city_delivery_line(
    *,
    city: str,
    rate: float,
    currency: str,
) -> str:
    fee = format_currency(rate, currency)
    return f"Delivery to {city}: {fee} flat rate per order (verified with Kapruka)"


_PERISHABLE_GIFT_RE = re.compile(
    r"\b(?:cake|cakes|flower|flowers|rose|roses|bouquet|fruit|chocolate|chocolates|"
    r"gift|gifts|hamper|hampers)\b",
    re.I,
)


def _reply_has_verified_delivery_fee(
    reply_text: str,
    *,
    rate: float | None = None,
    currency: str = "LKR",
) -> bool:
    if "verified with Kapruka" in reply_text:
        return True
    return rate is not None and format_currency(rate, currency) in reply_text


def _turn_implies_perishable_gift(
    user_message: str,
    *,
    session_product_focus: str | None = None,
) -> bool:
    if session_product_focus in ("cake", "flowers", "chocolate", "gift"):
        return True
    return bool(_PERISHABLE_GIFT_RE.search(user_message))


def _delivery_date_more_than_one_day_out(checked_date: str) -> bool:
    try:
        target = date.fromisoformat(checked_date)
    except ValueError:
        return False
    return (target - colombo_today()).days > 1


def _apply_perishable_delivery_honesty(
    reply_text: str,
    tool_trace: list[ToolInvocation] | None,
    *,
    user_message: str = "",
    session_product_focus: str | None = None,
    delivery_context_relevant: bool = True,
) -> tuple[str, str | None]:
    """Append verified delivery fee and perishable_warning; render delivery status partial."""
    if not delivery_context_relevant:
        return reply_text, None

    invocation = _last_check_delivery_invocation(tool_trace)
    if invocation is None:
        return reply_text, None

    delivery = invocation.get("result")
    if not isinstance(delivery, dict):
        return reply_text, None

    try:
        delivery_output = CheckDeliveryOutput.model_validate(delivery)
    except ValidationError:
        logger.warning("generate_response: invalid check_delivery payload for delivery honesty")
        return reply_text, None

    updated_reply = reply_text
    delivery_html: str | None = None

    if delivery_output.available:
        city = _canonical_city_from_check_delivery_invocation(invocation)
        if city and not _reply_has_verified_delivery_fee(
            updated_reply,
            rate=delivery_output.rate,
            currency=delivery_output.currency,
        ):
            if _is_city_only_check_delivery(invocation):
                fee_line = _build_verified_city_delivery_line(
                    city=city,
                    rate=delivery_output.rate,
                    currency=delivery_output.currency,
                )
            else:
                fee_line = _build_verified_delivery_fee_line(
                    city=city,
                    checked_date=delivery_output.checked_date,
                    rate=delivery_output.rate,
                    currency=delivery_output.currency,
                )
            updated_reply = f"{updated_reply}\n\n{fee_line}".strip()
        if not _is_city_only_check_delivery(invocation):
            delivery_html = render_delivery_date_status(result=delivery_output)

    warning = delivery_output.perishable_warning
    if (not isinstance(warning, str) or not warning.strip()) and _turn_implies_perishable_gift(
        user_message,
        session_product_focus=session_product_focus,
    ):
        checked_date = delivery_output.checked_date
        if (
            checked_date
            and not _is_city_only_check_delivery(invocation)
            and _delivery_date_more_than_one_day_out(checked_date)
        ):
            warning = (
                "Fresh cakes, flowers, and gift combos are best within a day or two of "
                "delivery. Your date is more than a day out — consider ordering closer to "
                "the event."
            )
    if isinstance(warning, str) and warning.strip():
        warning = warning.strip()
        dated_delivery = not _is_city_only_check_delivery(invocation)
        should_append_warning = warning not in updated_reply
        if should_append_warning:
            updated_reply = f"{updated_reply}\n\n{warning}".strip()
        if delivery_html is None and (
            dated_delivery or (warning and _is_city_only_check_delivery(invocation))
        ):
            delivery_html = render_delivery_date_status(result=delivery_output)

    return updated_reply, delivery_html


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


def _build_verified_dated_delivery_reply(
    *,
    city: str,
    checked_date: str,
    rate: float,
    currency: str,
) -> str:
    fee = format_currency(rate, currency)
    friendly_date = format_delivery_date_friendly(checked_date)
    return (
        f"Yes, we can deliver to {city} on {friendly_date}. "
        f"Delivery fee is {fee}."
    )


def _apply_verified_dated_delivery_template(
    reply_text: str,
    tool_trace: list[ToolInvocation] | None,
) -> str:
    """Use verified dated-delivery copy when check_delivery ran with a date."""
    invocation = _last_check_delivery_invocation(tool_trace)
    if invocation is None or _is_city_only_check_delivery(invocation):
        return reply_text
    delivery = invocation.get("result")
    if not isinstance(delivery, dict) or not delivery.get("available"):
        return reply_text
    city = _canonical_city_from_check_delivery_invocation(invocation)
    checked_date = delivery.get("checked_date")
    rate = delivery.get("rate")
    delivery_currency = delivery.get("currency") or "LKR"
    if (
        not city
        or not isinstance(checked_date, str)
        or not isinstance(rate, (int, float))
    ):
        return reply_text
    fee_label = format_currency(float(rate), str(delivery_currency))
    if "verified with Kapruka" in reply_text or fee_label in reply_text:
        return reply_text
    return _build_verified_dated_delivery_reply(
        city=city,
        checked_date=checked_date,
        rate=float(rate),
        currency=str(delivery_currency),
    )


def _is_cake_accessory(product: dict[str, Any]) -> bool:
    return is_cake_accessory(product)


def _filter_cake_search_products(
    products: list[dict[str, Any]],
    query: str | None,
) -> list[dict[str, Any]]:
    """Drop non-cake items and baking accessories when the search q targets cakes."""
    if not _is_cake_search_query(query):
        return products
    cake_products = [
        product
        for product in products
        if _is_likely_cake_product(product) and not _is_cake_accessory(product)
    ]
    return filter_cake_accessories(cake_products)


def _curated_search_results(search_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = search_payload.get("results")
    if not isinstance(raw_results, list):
        return []
    products = [
        item
        for item in raw_results
        if isinstance(item, dict) and item.get("id") and item.get("name")
    ]
    for product in products:
        name = product.get("name")
        if isinstance(name, str):
            product["name"] = normalize_catalog_text(name)
    query = _search_query_from_payload(search_payload)
    return _filter_cake_search_products(products, query)


def _discovery_curation_query(
    search_payload: dict[str, Any],
    *,
    user_message: str,
    session_search_query: str | None = None,
) -> str:
    if (
        is_budget_refinement_message(user_message)
        and isinstance(session_search_query, str)
        and session_search_query.strip()
    ):
        return session_search_query.strip()
    query = _search_query_from_payload(search_payload)
    return user_message.strip() or (query or "")


def _budget_curated_products(
    products: list[dict[str, Any]],
    *,
    query: str,
    budget_max: float | None,
    currency: str,
    graph_context_available: bool,
    hybrid_context: dict[str, Any] | None = None,
    session_product_focus: str | None = None,
    session_recipient_hint: str | None = None,
    session_occasion: str | None = None,
    strict_budget: bool = False,
) -> list[dict[str, Any]]:
    """Apply birthday/puja relevance and budget-aware carousel ordering."""
    return curate_carousel_products(
        products,
        query=query,
        budget_max=budget_max,
        currency=currency,
        graph_context_available=graph_context_available,
        hybrid_context=hybrid_context,
        session_product_focus=session_product_focus,
        session_recipient_hint=session_recipient_hint,
        session_occasion=session_occasion,
        strict_budget=strict_budget,
    )


def _refine_last_search_kwargs(
    *,
    budget_max: float,
    currency: str,
    user_message: str,
    session_product_focus: str | None,
    session_search_query: str | None = None,
    session_recipient_hint: str | None = None,
    hybrid_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "budget_max": budget_max,
        "currency": currency,
        "session_product_focus": session_product_focus,
        "session_search_query": session_search_query,
        "session_recipient_hint": session_recipient_hint,
        "user_message": user_message,
        "hybrid_context": hybrid_context,
    }


def _enrich_product_display_price(product: dict[str, Any]) -> dict[str, Any]:
    """Add display_price for LLM prose synthesis."""
    enriched = dict(product)
    raw_price = product.get("price")
    if isinstance(raw_price, dict):
        amount = raw_price.get("amount")
        currency = raw_price.get("currency") or "LKR"
        if isinstance(amount, (int, float)):
            enriched["display_price"] = format_currency(float(amount), str(currency))
    return enriched


def _cap_search_products_for_llm_context(
    tool_results: dict[str, Any] | None,
    *,
    limit: int = _LLM_CONTEXT_PRODUCT_LIMIT,
    budget_max: float | None = None,
    currency: str = "LKR",
    user_message: str = "",
    graph_context_available: bool = False,
    hybrid_context: dict[str, Any] | None = None,
    session_product_focus: str | None = None,
    session_search_query: str | None = None,
    session_recipient_hint: str | None = None,
    strict_budget: bool = False,
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

    curated = _budget_curated_products(
        _curated_search_results(search_payload),
        query=_discovery_curation_query(
            search_payload,
            user_message=user_message,
            session_search_query=session_search_query,
        ),
        budget_max=budget_max,
        currency=currency,
        graph_context_available=graph_context_available,
        hybrid_context=hybrid_context,
        session_product_focus=session_product_focus,
        session_recipient_hint=session_recipient_hint,
        strict_budget=strict_budget,
    )
    capped_results = [_enrich_product_display_price(product) for product in curated[:limit]]
    if capped_results == raw_results:
        return tool_results

    capped = dict(tool_results)
    capped_search = dict(search_payload)
    capped_search["results"] = capped_results
    capped[SEARCH_PRODUCTS_TOOL] = capped_search
    return capped


def _budget_prompt_line(budget_max: float | None, currency: str) -> str:
    if budget_max is None or budget_max <= 0:
        return ""
    return f"Customer budget cap: {format_currency(budget_max, currency)}.\n"


def _session_context_prompt_lines(
    *,
    session_search_query: str | None,
    session_occasion: str | None,
    session_recipient_hint: str | None,
    user_message: str,
) -> str:
    if not is_budget_refinement_message(user_message):
        return ""
    lines: list[str] = []
    if isinstance(session_search_query, str) and session_search_query.strip():
        lines.append(f"Session topic: {session_search_query.strip()}")
    if isinstance(session_occasion, str) and session_occasion.strip():
        lines.append(f"Occasion: {session_occasion.strip()}")
    if isinstance(session_recipient_hint, str) and session_recipient_hint.strip():
        lines.append(f"Recipient: {session_recipient_hint.strip()}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def _build_user_prompt(
    user_message: str,
    tool_results: dict[str, Any] | None,
    *,
    budget_max: float | None = None,
    currency: str = "LKR",
    session_search_query: str | None = None,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
) -> str:
    """Combine user turn and MCP payload for response synthesis."""
    context = _format_tool_results_context(tool_results)
    budget_line = _budget_prompt_line(budget_max, currency)
    session_line = _session_context_prompt_lines(
        session_search_query=session_search_query,
        session_occasion=session_occasion,
        session_recipient_hint=session_recipient_hint,
        user_message=user_message,
    )
    return (
        f"Customer message:\n{user_message}\n\n"
        f"{session_line}"
        f"{budget_line}"
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
    delivery_context_relevant: bool = True,
) -> str:
    """Blocking Gemini call; run via asyncio.to_thread from generate_response."""
    instruction = system_instruction or build_response_system_instruction(
        intent_metadata,
        zep_memory_facts=zep_memory_facts,
        intent=intent,
        delivery_context_relevant=delivery_context_relevant,
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


def _carousel_strict_budget(
    user_message: str,
    budget_max: float | None,
    *,
    state: AgentState | None = None,
) -> bool:
    if budget_max is None or budget_max <= 0:
        return False
    from lib.chat.intent_heuristics import has_explicit_budget_constraint

    pivot_meta = (state or {}).get("intent_metadata") or {}
    topic_pivot = bool(
        isinstance(pivot_meta, dict) and pivot_meta.get("topic_pivot"),
    )
    session_budget = (state or {}).get("session_budget_max")
    return has_explicit_budget_constraint(
        user_message,
        session_budget if isinstance(session_budget, (int, float)) else None,
        topic_pivot=topic_pivot,
    )


def _prepend_situational_empathy(
    reply_text: str,
    intent_metadata: IntentMetadata | None,
) -> str:
    if not intent_metadata or not intent_metadata.get("is_situational"):
        return reply_text
    head = reply_text.strip().lower()[:120]
    if any(
        phrase in head
        for phrase in ("sorry", "hear that", "heartbroken", "going through")
    ):
        return reply_text
    return f"I'm sorry to hear you're going through this. {reply_text.strip()}"


def _prepend_budget_confirmation(
    reply_text: str,
    intent_metadata: IntentMetadata | None,
    *,
    budget_max: float | None,
    currency: str,
) -> str:
    if not intent_metadata or not intent_metadata.get("budget_confirmation_pending"):
        return reply_text
    if budget_max is None or budget_max <= 0:
        return reply_text
    cap = format_currency(budget_max, currency)
    if "keeping under" in reply_text.lower() or cap.lower() in reply_text.lower():
        return reply_text
    return f"Still keeping under {cap}?\n\n{reply_text.strip()}"


def _emit_synthesis_status() -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    if writer is not None:
        writer({"type": "status", "message": PUTTING_TOGETHER_RECOMMENDATIONS})


def extract_search_products(
    tool_results: dict[str, Any] | None,
    *,
    budget_max: float | None = None,
    currency: str = "LKR",
    user_message: str = "",
    graph_context_available: bool = False,
    hybrid_context: dict[str, Any] | None = None,
    session_product_focus: str | None = None,
    session_search_query: str | None = None,
    session_recipient_hint: str | None = None,
    session_occasion: str | None = None,
    strict_budget: bool = False,
    last_search_products: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return curated product dicts from kapruka_search_products tool_results, if any."""
    if not tool_results:
        return []

    search_payload = tool_results.get(SEARCH_PRODUCTS_TOOL)
    if not isinstance(search_payload, dict):
        return []

    products = _curated_search_results(search_payload)
    products = _budget_curated_products(
        products,
        query=_discovery_curation_query(
            search_payload,
            user_message=user_message,
            session_search_query=session_search_query,
        ),
        budget_max=budget_max,
        currency=currency,
        graph_context_available=graph_context_available,
        hybrid_context=hybrid_context,
        session_product_focus=session_product_focus,
        session_recipient_hint=session_recipient_hint,
        session_occasion=session_occasion,
        strict_budget=strict_budget,
    )
    if (
        session_product_focus
        and products
        and not carousel_focus_guard(products, session_product_focus)
        and last_search_products
        and budget_max is not None
        and budget_max > 0
    ):
        refined = refine_last_search_by_budget(
            last_search_products,
            budget_max=budget_max,
            currency=currency,
            session_product_focus=session_product_focus,
            session_search_query=session_search_query,
            session_recipient_hint=session_recipient_hint,
            user_message=user_message,
            hybrid_context=hybrid_context,
        )
        if refined:
            return refined
    return products


def _build_cart_assistant_message(action: dict[str, Any]) -> str | None:
    """Synthesize add-to-cart confirmation or error copy from cart_action_result."""
    status = action.get("status")
    if status == "clarify":
        question = action.get("clarifying_question")
        return str(question).strip() if isinstance(question, str) and question.strip() else None
    if status == "error":
        message = action.get("message")
        return str(message).strip() if isinstance(message, str) and message.strip() else None
    if status != "added":
        return None
    name = str(action.get("product_name") or "item")
    quantity = action.get("quantity")
    if action.get("merged") and isinstance(quantity, int):
        return f"Updated your cart — {name} is now quantity {quantity}."
    return f"Added {name} to your cart."


def _build_cart_oob_html(action: dict[str, Any], *, currency: str) -> str | None:
    """Render OOB cart panel swap when a cart line was added."""
    if action.get("status") != "added":
        return None
    raw_items = action.get("cart_items")
    if not isinstance(raw_items, list) or not raw_items:
        return None
    items = [StoredCartItem.model_validate(row) for row in raw_items if isinstance(row, dict)]
    if not items:
        return None
    return render_cart_partial_oob(items=items, currency=currency)


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
        return f"'{name}' for {format_currency(float(amount), currency)}, {stock_note}"
    return f"'{name}', {stock_note}"


def _apply_artificial_floral_honesty(
    reply_text: str,
    products: list[dict[str, Any]],
    *,
    user_message: str = "",
) -> str:
    """Prepend proactive artificial-floral disclosure when flowers requests surface silk picks."""
    note = artificial_floral_note_for_picks(products, user_message=user_message)
    if note is None:
        return reply_text
    if reply_already_discloses_artificial_floral(reply_text):
        return reply_text
    return f"{note}\n\n{reply_text}".strip()


def _build_discovery_template_reply(
    products: list[dict[str, Any]],
    *,
    user_message: str = "",
) -> str:
    """Deterministic assistant copy from MCP search results (no Gemini)."""
    if not products:
        return ""
    picks = products[:3]
    lines = [_format_product_line(product) for product in picks]
    opener = "Here are a few thoughtful Kapruka picks:"
    if len(lines) == 1:
        body = f"{opener} {lines[0]}."
    elif len(lines) == 2:
        body = f"{opener} {lines[0]}, and {lines[1]}."
    else:
        body = f"{opener} {lines[0]}, {lines[1]}, and {lines[2]}."
    note = artificial_floral_note_for_picks(picks, user_message=user_message)
    if note:
        return f"{note}\n\n{body}"
    return body


def build_products_carousel_html(
    tool_results: dict[str, Any] | None,
    *,
    budget_max: float | None = None,
    currency: str = "LKR",
    user_message: str = "",
    graph_context_available: bool = False,
    hybrid_context: dict[str, Any] | None = None,
    session_product_focus: str | None = None,
    last_search_products: list[dict[str, Any]] | None = None,
    last_visible_products: list[dict[str, Any]] | None = None,
    visible_products: list[dict[str, Any]] | None = None,
    allow_stale_fallback: bool = True,
) -> str | None:
    """Render product carousel partial when search_products returned results."""
    products = visible_products
    if products is None:
        products = extract_search_products(
            tool_results,
            budget_max=budget_max,
            currency=currency,
            user_message=user_message,
            graph_context_available=graph_context_available,
            hybrid_context=hybrid_context,
            session_product_focus=session_product_focus,
            last_search_products=last_search_products,
        )
    if not products and allow_stale_fallback and last_visible_products:
        products = last_visible_products
    if not products and allow_stale_fallback and last_search_products:
        if budget_max is not None and budget_max > 0:
            refined = refine_last_search_by_budget(
                last_search_products,
                budget_max=budget_max,
                currency=currency,
                session_product_focus=session_product_focus,
                user_message=user_message,
                hybrid_context=hybrid_context,
            )
            if refined:
                products = refined
        if not products:
            products = last_search_products
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
    if step == "delivery_city":
        return (
            "Which Kapruka delivery city should we send this to? "
            "For example: Colombo 03, Kandy, or Galle."
        )
    if step == "delivery_date":
        city = checkout.get("delivery_city") or "your city"
        return (
            f"When should we deliver to {city}? "
            "Share a date (for example next Saturday or YYYY-MM-DD)."
        )
    if step == "recipient":
        return (
            "Who should receive this gift? Share their name and Sri Lankan mobile number "
            "(for example Amaya, 0771234567)."
        )
    if step == "sender":
        return (
            "What name should appear on the gift card? "
            "Say 'anonymous' if you prefer not to include your name."
        )
    if step == "review":
        return (
            "Please review your order details below. "
            "Say 'confirm' when you're ready to pay securely."
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
    delivery_status_html: str | None = None,
    rate_limit_banner_html: str | None = None,
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
        delivery_status_html=delivery_status_html,
        rate_limit_banner_html=rate_limit_banner_html,
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

    off_topic_meta = dict(state.get("intent_metadata") or {})
    if state.get("intent") == "general" and off_topic_meta.get("is_off_topic"):
        redirect_kind = off_topic_meta.get("redirect_kind")
        if redirect_kind == "impossible_product":
            reply = build_impossible_product_redirect(impossible_request_subject(user_message))
        else:
            reply = build_off_topic_redirect_message(off_topic_topic(user_message))
        return {
            "response_html": render_assistant_html(reply),
            "assistant_message": reply,
        }

    if _is_general_welcome_path(state):
        if is_farewell_message(user_message):
            farewell = build_farewell_message()
            return {
                "response_html": render_assistant_html(farewell),
                "assistant_message": farewell,
            }
        welcome = build_general_welcome_message()
        return {
            "response_html": render_assistant_html(welcome),
            "assistant_message": welcome,
        }

    clarifying_question = state.get("agent_clarifying_question")
    exit_reason = state.get("agent_loop_exit_reason")
    intent_metadata = state.get("intent_metadata") or {}
    situational_with_products = (
        isinstance(intent_metadata, dict)
        and intent_metadata.get("is_situational")
        and _turn_search_has_products(state.get("tool_trace"))
    )
    show_clarifying = (
        isinstance(clarifying_question, str)
        and clarifying_question.strip()
        and not situational_with_products
        and (
            exit_reason == "ask_user"
            or (
                exit_reason is None
                and not _turn_has_fresh_search(state.get("tool_trace"))
            )
        )
    )
    if show_clarifying:
        question = clarifying_question.strip()
        tool_trace = state.get("tool_trace")
        delivery_context_relevant = is_delivery_context_relevant_turn(dict(state), user_message)
        question, delivery_status_html = _apply_perishable_delivery_honesty(
            question,
            tool_trace,
            user_message=user_message,
            session_product_focus=state.get("session_product_focus"),
            delivery_context_relevant=delivery_context_relevant,
        )
        return {
            "response_html": render_assistant_html(
                question,
                delivery_status_html=delivery_status_html,
            ),
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
        error_code = agent_tool_error.get("error") or _error_code_from_tool_trace(
            state.get("tool_trace"),
            tool_name,
        )
        order_number = extract_order_number(user_message)
        error_reply = build_agent_tool_error_message(
            tool=tool_name,
            raw_message=raw_message,
            error_code=error_code,
            order_number=order_number,
            reference_kind=(classify_order_reference(order_number) if order_number else None),
        )
        last_search = list(state.get("last_search_products") or [])
        if is_product_detail_turn(user_message):
            matched = match_product_from_last_search(user_message, last_search)
            if matched is not None:
                detail = summarize_product_from_carousel(matched)
                error_reply = f"{error_reply}\n\nFrom our earlier results: {detail}"
        tool_trace = state.get("tool_trace")
        if is_delivery_fee_question(user_message) and _tool_trace_has_check_delivery(tool_trace):
            delivery_context_relevant = is_delivery_context_relevant_turn(dict(state), user_message)
            reply_text, delivery_status_html = _apply_perishable_delivery_honesty(
                error_reply,
                tool_trace,
                user_message=user_message,
                session_product_focus=state.get("session_product_focus"),
                delivery_context_relevant=delivery_context_relevant,
            )
            return {
                "response_html": render_assistant_html(
                    reply_text,
                    delivery_status_html=delivery_status_html,
                ),
                "assistant_message": reply_text,
            }
        error_code = agent_tool_error.get("error") or _error_code_from_tool_trace(
            state.get("tool_trace"),
            tool_name,
        )
        allow_stale = not _turn_has_fresh_search(state.get("tool_trace"))
        if error_code in ("429", "rate_limit_exceeded"):
            allow_stale = True
        products_html = build_products_carousel_html(
            _resolve_effective_tool_results(state),
            budget_max=state.get("session_budget_max"),
            currency=state.get("currency") or "LKR",
            user_message=user_message,
            graph_context_available=has_graph_hybrid_context(state.get("hybrid_context") or {}),
            hybrid_context=state.get("hybrid_context") or {},
            session_product_focus=state.get("session_product_focus"),
            last_search_products=last_search or None,
            allow_stale_fallback=allow_stale,
        )
        rate_limit_banner = _rate_limit_banner_html(agent_tool_error)
        return {
            "response_html": render_assistant_html(
                error_reply,
                products_html=products_html,
                rate_limit_banner_html=rate_limit_banner,
            ),
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

        order_number = extract_order_number(user_message)
        track_error = tracking_error_from_tool_results(tool_results)
        if track_error:
            failure_reply = build_tracking_failure_message(
                order_number=order_number,
                reference_kind=(
                    classify_order_reference(order_number) if order_number else "unknown"
                ),
                error_code=track_error.get("error"),
                raw_message=track_error.get("message"),
            )
            return {
                "response_html": render_assistant_html(failure_reply),
                "assistant_message": failure_reply,
            }

        if not order_number:
            missing_number_reply = build_missing_tracking_number_message(user_message)
            return {
                "response_html": render_assistant_html(missing_number_reply),
                "assistant_message": missing_number_reply,
            }

        failure_reply = build_tracking_failure_message(
            order_number=order_number,
            reference_kind=classify_order_reference(order_number),
            error_code="order_not_found",
            raw_message=None,
        )
        return {
            "response_html": render_assistant_html(failure_reply),
            "assistant_message": failure_reply,
        }

    if state.get("intent") == "cart":
        action = dict(state.get("cart_action_result") or {})
        cart_reply = _build_cart_assistant_message(action)
        if not cart_reply:
            cart_reply = "I couldn't add that — try naming the product."
        cart_oob = _build_cart_oob_html(
            action,
            currency=state.get("currency") or "LKR",
        )
        assistant_html = render_assistant_html(cart_reply)
        response_html = f"{cart_oob}{assistant_html}" if cart_oob else assistant_html
        return {
            "response_html": response_html,
            "assistant_message": cart_reply,
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
                can_refine_from_last_search = bool(
                    is_budget_refinement_message(user_message)
                    and state.get("last_search_products")
                )
                if not can_refine_from_last_search:
                    empty_reply = build_empty_search_reply(
                        broaden_attempted=bool(state.get("search_broaden_applied")),
                    )
                    return {
                        "response_html": render_assistant_html(empty_reply),
                        "assistant_message": empty_reply,
                    }

    currency = state.get("currency") or "LKR"
    metadata = dict(state.get("intent_metadata") or {})
    session_budget = state.get("session_budget_max")
    turn_budget = metadata.get("budget_max")
    budget_max: float | None = None
    if _session_budget_applies(state, user_message):
        if isinstance(session_budget, (int, float)) and session_budget > 0:
            budget_max = float(session_budget)
        elif isinstance(turn_budget, (int, float)) and turn_budget > 0:
            budget_max = float(turn_budget)
    graph_context_available = has_graph_hybrid_context(state.get("hybrid_context") or {})
    hybrid_context = state.get("hybrid_context") or {}
    session_product_focus = state.get("session_product_focus")
    delivery_context_relevant = is_delivery_context_relevant_turn(dict(state), user_message)
    last_search_products = list(state.get("last_search_products") or [])
    last_visible_products = list(state.get("last_visible_products") or [])
    pivot_meta = state.get("intent_metadata") or {}
    topic_pivot = bool(pivot_meta.get("topic_pivot")) if isinstance(pivot_meta, dict) else False
    fresh_search = _turn_has_fresh_search(state.get("tool_trace"))
    allow_stale_fallback = not topic_pivot and not fresh_search

    if is_product_detail_turn(user_message):
        matched = match_product_from_last_search(
            user_message,
            state.get("last_search_products"),
        )
        get_payload = (tool_results or {}).get(GET_PRODUCT_TOOL)
        detail_reply: str | None = None
        if (
            isinstance(get_payload, dict)
            and not get_payload.get("error")
            and get_payload.get("name")
        ):
            detail_reply = summarize_product_from_carousel(get_payload)
        elif matched is not None:
            detail_reply = summarize_product_from_carousel(matched)
        if detail_reply:
            tool_trace = state.get("tool_trace")
            detail_reply, delivery_status_html = _apply_perishable_delivery_honesty(
                detail_reply,
                tool_trace,
                user_message=user_message,
                session_product_focus=session_product_focus,
                delivery_context_relevant=delivery_context_relevant,
            )
            products_html = build_products_carousel_html(
                tool_results,
                budget_max=budget_max,
                currency=currency,
                user_message=user_message,
                graph_context_available=graph_context_available,
                hybrid_context=hybrid_context,
                session_product_focus=session_product_focus,
                last_search_products=state.get("last_search_products"),
            )
            return {
                "response_html": render_assistant_html(
                    detail_reply,
                    products_html=products_html,
                    delivery_status_html=delivery_status_html,
                ),
                "assistant_message": detail_reply,
            }

    visible_products: list[dict[str, Any]] | None = None
    if (
        is_budget_refinement_message(user_message)
        and last_search_products
        and budget_max is not None
        and budget_max > 0
    ):
        refined = refine_last_search_by_budget(
            last_search_products,
            budget_max=budget_max,
            currency=currency,
            session_product_focus=session_product_focus,
            session_search_query=state.get("session_search_query"),
            session_recipient_hint=state.get("session_recipient_hint"),
            user_message=user_message,
            hybrid_context=hybrid_context,
        )
        if refined:
            visible_products = refined

    if visible_products is None:
        strict_budget = _carousel_strict_budget(user_message, budget_max, state=state)
        visible_products = extract_search_products(
            tool_results,
            budget_max=budget_max,
            currency=currency,
            user_message=user_message,
            graph_context_available=graph_context_available,
            hybrid_context=hybrid_context,
            session_product_focus=session_product_focus,
            session_search_query=state.get("session_search_query"),
            session_recipient_hint=state.get("session_recipient_hint"),
            session_occasion=state.get("session_occasion"),
            strict_budget=strict_budget,
            last_search_products=last_search_products or None,
        )
        if (
            visible_products
            and not carousel_focus_guard(visible_products, session_product_focus)
            and budget_max is not None
            and budget_max > 0
        ):
            refined = refine_last_search_by_budget(
                last_search_products,
                budget_max=budget_max,
                currency=currency,
                session_product_focus=session_product_focus,
                session_search_query=state.get("session_search_query"),
                session_recipient_hint=state.get("session_recipient_hint"),
                user_message=user_message,
                hybrid_context=hybrid_context,
            )
            if refined:
                visible_products = refined

    products_html = build_products_carousel_html(
        tool_results,
        budget_max=budget_max,
        currency=currency,
        user_message=user_message,
        graph_context_available=graph_context_available,
        hybrid_context=hybrid_context,
        session_product_focus=session_product_focus,
        last_search_products=last_search_products or None,
        last_visible_products=last_visible_products or None,
        visible_products=visible_products,
        allow_stale_fallback=allow_stale_fallback,
    )

    effective_tool_results = _suppress_delivery_tool_results(
        tool_results,
        delivery_context_relevant=delivery_context_relevant,
    )
    strict_budget = _carousel_strict_budget(user_message, budget_max, state=state)
    client = genai_client
    model = select_model(state)
    _emit_synthesis_status()
    user_prompt = _build_user_prompt(
        user_message,
        _cap_search_products_for_llm_context(
            effective_tool_results,
            budget_max=budget_max,
            currency=currency,
            user_message=user_message,
            graph_context_available=graph_context_available,
            hybrid_context=hybrid_context,
            session_product_focus=session_product_focus,
            session_search_query=state.get("session_search_query"),
            session_recipient_hint=state.get("session_recipient_hint"),
            strict_budget=strict_budget,
        ),
        budget_max=budget_max,
        currency=currency,
        session_search_query=state.get("session_search_query"),
        session_occasion=state.get("session_occasion"),
        session_recipient_hint=state.get("session_recipient_hint"),
    )

    zep_memory_facts = state.get("zep_memory_facts")
    if zep_memory_facts:
        zep_memory_facts = scope_memory_facts_for_turn(
            zep_memory_facts,
            user_message,
            is_budget_refinement=is_budget_refinement_message(user_message),
        )
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
            delivery_context_relevant=delivery_context_relevant,
        )
    except Exception as exc:
        if not is_resource_exhausted(exc):
            raise
        reply_text = _build_discovery_template_reply(visible_products, user_message=user_message)
        logger.warning(
            "generate_response: Gemini rate limited; template fallback (%d products)",
            len(visible_products),
            exc_info=True,
        )

    if not reply_text:
        template = _build_discovery_template_reply(
            visible_products,
            user_message=user_message,
        )
        reply_text = template or "I could not generate a response. Please try again."

    reply_text = normalize_catalog_text(reply_text)
    # Prepend Sunday-ambiguity clarification before other post-processing
    if isinstance(intent_metadata, dict) and intent_metadata.get("delivery_date_ambiguous"):
        clarification = intent_metadata.get("delivery_date_clarification") or ""
        if clarification and clarification.lower() not in reply_text.lower():
            reply_text = f"{clarification}\n\n{reply_text.strip()}"
    reply_text = _prepend_budget_confirmation(
        reply_text,
        intent_metadata,
        budget_max=budget_max,
        currency=currency,
    )
    reply_text = _prepend_situational_empathy(reply_text, intent_metadata)
    tool_trace = state.get("tool_trace")
    reply_text = delivery_claim_guard(
        reply_text,
        tool_trace,
        user_message=user_message,
        delivery_city_status=state.get("delivery_city_status"),
        delivery_city_confirmed=bool(state.get("session_delivery_city_confirmed")),
        delivery_context_relevant=delivery_context_relevant,
    )
    if delivery_context_relevant:
        reply_text = _apply_verified_dated_delivery_template(reply_text, tool_trace)
    reply_text = carousel_consistency_guard(
        reply_text,
        visible_products,
        user_message=user_message,
        budget_max=budget_max,
        currency=currency,
        strict_budget=strict_budget,
    )
    reply_text = _apply_artificial_floral_honesty(
        reply_text,
        visible_products,
        user_message=user_message,
    )
    reply_text, delivery_status_html = _apply_perishable_delivery_honesty(
        reply_text,
        tool_trace,
        user_message=user_message,
        session_product_focus=session_product_focus,
        delivery_context_relevant=delivery_context_relevant,
    )

    logger.info(
        "generate_response: rendered assistant reply (%d chars, carousel=%s)",
        len(reply_text),
        bool(products_html),
    )
    updates: dict[str, Any] = {
        "response_html": render_assistant_html(
            reply_text,
            products_html=products_html,
            delivery_status_html=delivery_status_html,
        ),
        "assistant_message": reply_text,
    }
    if visible_products:
        updates["last_visible_products"] = visible_products
    if topic_pivot:
        updates["last_visible_products"] = visible_products or None
        updates["last_search_products"] = visible_products or None
    if isinstance(intent_metadata, dict) and (
        intent_metadata.get("budget_confirmation_pending")
        or intent_metadata.get("delivery_date_ambiguous")
    ):
        cleared = dict(intent_metadata)
        cleared["budget_confirmation_pending"] = False
        cleared["delivery_date_ambiguous"] = False
        updates["intent_metadata"] = cast(IntentMetadata, cleared)
    return updates
