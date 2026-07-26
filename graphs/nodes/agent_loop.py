"""Bounded ReAct agent loop node — planner loop and trace summarization."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal, TypedDict, cast
from urllib.parse import urlparse

from langgraph.config import get_stream_writer
from pydantic import BaseModel

from graphs.model_router import FLASH_MODEL
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, Intent, ToolInvocation
from lib.chat.delivery_dates import (
    delivery_date_clarifying_question,
    normalize_delivery_date,
)
from lib.chat.intent_heuristics import (
    has_explicit_budget_constraint,
    is_bare_category_pivot,
    is_budget_refinement_message,
    is_budgeted_gift_ideas_message,
)
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.product_curation import (
    apply_anniversary_curation,
    apply_birthday_cake_curation,
    apply_gift_curation,
    apply_puja_curation,
    apply_recipient_curation,
    carousel_focus_guard,
    demote_non_chocolate_for_chocolate_focus,
    demote_non_tea_for_tea_focus,
    demote_off_focus_products,
    filter_excluded_category_hints,
    filter_gift_noise_products,
    has_graph_hybrid_context,
    is_anniversary_occasion_intent,
    is_cake_accessory,
    is_flower_fruit_intent,
    product_price_amount,
    refine_last_search_by_budget,
)
from lib.chat.product_detail import (
    is_delivery_fee_question,
    is_product_detail_turn,
    is_valid_product_detail_payload,
    match_product_from_last_search,
    normalize_resolved_product,
)
from lib.chat.query_preprocessor import (
    _has_perishable_gift_intent,
    is_delivery_context_relevant_turn,
    should_defer_delivery_date,
)
from lib.chat.request_specificity import is_delivery_only_inquiry
from lib.chat.search_broadening import apply_first_broaden
from lib.chat.status_copy import SEARCHING_CATALOG, long_search_status_message
from lib.chat.support_faq import is_support_question
from lib.debug.trace import trace_agent_iteration
from lib.genai.completions import generate_content
from lib.genai.errors import is_rate_limited, is_transient_nim_error
from lib.kapruka.service import KaprukaService
from lib.kapruka.tool_executor import (
    canonical_tool_args_for_dedup,
    enrich_get_product_args,
    inject_currency,
    invoke_tool,
    normalize_planner_tool_args,
)
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.neo4j.hybrid_context import (
    anniversary_fallback_search_queries,
    budget_focus_fallback_queries,
    build_budget_refinement_search_args,
    build_discovery_search_args,
    enrich_message_with_session_slots,
    has_strong_hybrid_hints,
    is_birthday_cake_intent,
    is_broad_cakes_query,
    is_confident_discovery_turn,
    merge_planner_search_args,
    strip_location_from_search_query,
)
from lib.utils.timezone import colombo_today_iso
from lib.zep.memory import format_memory_facts_block, scope_memory_facts_for_turn

logger = logging.getLogger(__name__)


def _tool_trace_entry(name: str, args: dict[str, Any], result: Any) -> ToolInvocation:
    return cast(ToolInvocation, {"name": name, "args": args, "result": result})


MAX_ITERATIONS = 3
GRAPH_HINTS_MAX_ITERATIONS = 2
UTILITY_GENERAL_MAX_ITERATIONS = 2
CONFIDENT_DISCOVERY_MAX_ITERATIONS = 1
BUDGETED_GIFT_SEARCH_CONCURRENCY = 2
BUDGET_REFINEMENT_IN_MEMORY_MIN = 1
PLANNER_MODEL = FLASH_MODEL

ALLOWED_PLANNER_TOOLS: frozenset[str] = frozenset(
    {
        SEARCH_PRODUCTS_TOOL,
        GET_PRODUCT_TOOL,
        LIST_CATEGORIES_TOOL,
        LIST_CITIES_TOOL,
        CHECK_DELIVERY_TOOL,
    },
)

_TOOL_STATUS_MESSAGES: dict[str, str] = {
    SEARCH_PRODUCTS_TOOL: SEARCHING_CATALOG,
    CHECK_DELIVERY_TOOL: "Checking delivery…",
    LIST_CITIES_TOOL: "Listing delivery cities…",
    LIST_CATEGORIES_TOOL: "Browsing categories…",
    GET_PRODUCT_TOOL: "Fetching product details…",
}
_DEFAULT_STATUS_MESSAGE = SEARCHING_CATALOG

PLANNER_SEARCH_RESULT_LIMIT = 5
PLANNER_CATEGORY_NODE_LIMIT = 10

_SEARCH_PRODUCT_FIELDS = frozenset({"id", "name", "price", "in_stock", "stock_level"})
_GET_PRODUCT_FIELDS = frozenset({"id", "name", "price", "in_stock", "stock_level"})

PLANNER_SYSTEM_INSTRUCTION = """You are the Kapruka gift shopping assistant catalog planner.

Decide the next step to gather catalog facts for the customer's request.

Return structured JSON with:
- action: call_tool | finish | ask_user
- tool_name: MCP tool name when action is call_tool (otherwise null)
- tool_args: tool arguments object when action is call_tool (otherwise null)
- rationale: for ask_user, a short customer-facing clarifying question; otherwise a brief trace note

Allowed tools only:
- kapruka_search_products (tool_args must include string field q, not query)
- kapruka_get_product
- kapruka_list_categories
- kapruka_check_delivery
- kapruka_list_delivery_cities

Never call kapruka_track_order.

Finish rule: If products matching the user's core request have been retrieved,
action MUST be finish. Do not run auxiliary category browsing or extra searches
unless the user explicitly requested them.

Broad gifts example: Customer says "show me gifts" or "some gifts" with no
occasion, recipient, or budget named → action MUST be ask_user to learn who the
gift is for or what occasion before running kapruka_search_products.

Budgeted gift queries: When the customer names a budget (e.g. "gift ideas under
Rs. 5,000") with no product topic in session → run two kapruka_search_products calls:
(1) q="gift voucher" with max_price from their budget, then (2) q="gift hamper" or
q="chocolates gift" with the same max_price. Merge voucher and physical gift results
before finish.

When session shopping focus or session_search_query is set, budget-only turns must
reuse that product context with max_price — do not switch to gift vouchers.

Display currency (authoritative): use the session currency from the user prompt
for all price filters; do not ask the customer to choose LKR vs USD when the UI
session currency is set.

When a delivery date is already resolved in the user prompt, do not ask_user for
the date again — use kapruka_check_delivery with that date when delivery is needed.

Prior session facts (Zep) are conversational context only — not catalog truth.
Hybrid context hints and preferences are soft hints; the explicit user message
always wins over inferred preferences.

Prior tool iterations in the user prompt are summarized summaries only — never
assume full catalog payloads from summaries alone.

On every step also set refined_intent:
- discovery: browsing, searching, or finding gifts and products
- general: greetings, thanks, FAQ, or unclear/off-topic messages

Shopping turns that need no catalog tools should use refined_intent general with
action finish.

Situational empathy turns: When the customer shares emotional context (breakup,
condolence, apology) AND names a product type (flowers, roses, bouquet), action
MUST be call_tool with kapruka_search_products — not ask_user. Search first, then
finish so the response can pair empathy with product options.
"""


class AgentPlannerStep(BaseModel):
    """Structured Gemini planner response for one agent-loop iteration."""

    action: Literal["call_tool", "finish", "ask_user"]
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    rationale: str = ""
    refined_intent: Literal["discovery", "general"] | None = None


class PlannerTraceEntry(TypedDict):
    """Planner-safe view of a prior tool invocation — summary only, never full result."""

    name: str
    args: dict[str, Any]
    summary: Any


def _summarize_error(result: dict[str, Any]) -> dict[str, Any]:
    return {"error": result.get("error"), "message": result.get("message")}


def _category_id_from_url(url: str) -> str:
    """Derive a stable category id from a Kapruka category URL path segment."""
    path = urlparse(url).path.rstrip("/")
    if not path:
        return url
    return path.rsplit("/", 1)[-1] or url


def _flatten_category_nodes(
    nodes: list[dict[str, Any]],
    *,
    limit: int = PLANNER_CATEGORY_NODE_LIMIT,
) -> list[dict[str, str]]:
    """Walk category tree depth-first, collecting name + id pairs up to limit."""
    collected: list[dict[str, str]] = []

    def walk(items: list[dict[str, Any]]) -> None:
        for node in items:
            if len(collected) >= limit:
                return
            if not isinstance(node, dict):
                continue
            name = node.get("name")
            if isinstance(name, str) and name:
                url = node.get("url")
                category_id = _category_id_from_url(url) if isinstance(url, str) and url else name
                collected.append({"name": name, "id": category_id})
            children = node.get("children")
            if isinstance(children, list) and children:
                walk(children)

    walk(nodes)
    return collected


def _summarize_search_products(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return {"results": []}
    summarized: list[dict[str, Any]] = []
    for product in raw_results[:PLANNER_SEARCH_RESULT_LIMIT]:
        if not isinstance(product, dict):
            continue
        summarized.append({key: product[key] for key in _SEARCH_PRODUCT_FIELDS if key in product})
    return {"results": summarized}


def _summarize_get_product(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    return {key: result[key] for key in _GET_PRODUCT_FIELDS if key in result}


def _summarize_list_categories(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    categories = result.get("categories")
    if not isinstance(categories, list):
        return {"categories": []}
    return {"categories": _flatten_category_nodes(categories)}


def _summarize_check_delivery(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    return {
        "city": result.get("city"),
        "deliverable": result.get("available"),
    }


def _summarize_list_delivery_cities(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    cities = result.get("cities")
    if not isinstance(cities, list):
        return {"cities": [], "total_matched": 0}
    names = [
        str(entry["name"])
        for entry in cities[:PLANNER_CATEGORY_NODE_LIMIT]
        if isinstance(entry, dict) and entry.get("name")
    ]
    return {
        "cities": names,
        "total_matched": result.get("total_matched"),
    }


def _summarize_for_planner(name: str, result: Any) -> Any:
    """Return a compact tool payload safe for planner prompt context.

    Full MCP results remain in ``tool_trace[].result`` for ``generate_response``.
    """
    if isinstance(result, dict) and "error" in result:
        return _summarize_error(result)
    if name == SEARCH_PRODUCTS_TOOL:
        return _summarize_search_products(result)
    if name == GET_PRODUCT_TOOL:
        return _summarize_get_product(result)
    if name == LIST_CATEGORIES_TOOL:
        return _summarize_list_categories(result)
    if name == CHECK_DELIVERY_TOOL:
        return _summarize_check_delivery(result)
    if name == LIST_CITIES_TOOL:
        return _summarize_list_delivery_cities(result)
    if isinstance(result, dict):
        return _summarize_error(result) if "error" in result else {"status": "ok"}
    return result


def build_planner_prior_iterations(
    tool_trace: list[ToolInvocation] | None,
) -> list[PlannerTraceEntry]:
    """Build planner-safe prior-iteration context — never includes full MCP payloads."""
    if not tool_trace:
        return []
    return [
        {
            "name": invocation["name"],
            "args": invocation["args"],
            "summary": _summarize_for_planner(invocation["name"], invocation["result"]),
        }
        for invocation in tool_trace
    ]


def format_planner_prior_iterations(tool_trace: list[ToolInvocation] | None) -> str:
    """Serialize prior tool iterations for injection into the planner system prompt."""
    entries = build_planner_prior_iterations(tool_trace)
    if not entries:
        return ""
    return json.dumps(entries, ensure_ascii=False)


def _resolve_budget_currency(state: AgentState) -> str | None:
    """Message-explicit budget currency wins over session display currency."""
    return state.get("session_budget_currency") or None


def _has_budget_query(state: AgentState) -> bool:
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    budget_max = intent_metadata.get("budget_max") or state.get("session_budget_max")
    return isinstance(budget_max, (int, float)) and budget_max > 0


def _inject_tool_currency(
    tool_name: str,
    args: dict[str, Any],
    state: AgentState,
    session_currency: str,
) -> dict[str, Any]:
    return inject_currency(
        tool_name,
        args,
        session_currency,
        budget_currency=_resolve_budget_currency(state),
    )


def _resolve_currency(state: AgentState) -> str:
    """Session currency wins; fall back to Zep hints then LKR."""
    hybrid_context = state.get("hybrid_context") or {}
    hints = hybrid_context.get("hints") or {}
    preferences = hybrid_context.get("preferences") or {}
    return state.get("currency") or hints.get("currency") or preferences.get("currency") or "LKR"


_CAKES_BROAD = re.compile(r"\bcakes?\b", re.I)
_BIRTHDAY_CAKE = re.compile(r"\bbirthday\s+cake", re.I)
_MOM_BIRTHDAY = re.compile(
    r"\b(?:mom|mother|mum|amma)\b.*\bbirthday\b|\bbirthday\b.*\b(?:mom|mother|mum|amma)\b",
    re.I,
)
_BROAD_GIFTS = re.compile(
    r"^(?:show me )?(?:some )?gifts?\s*[!.?]*$",
    re.I,
)
_GIFT_WORD = re.compile(r"\bgifts?\b", re.I)
_SHORT_CATEGORY_REPLY = re.compile(
    r"^(?:cakes?|flowers?|chocolates?|roses?|bouquets?)\s*[!.?]*$",
    re.I,
)
_FLOWERS_REQUEST = re.compile(r"\b(?:flower|flowers|rose|roses|bouquet|floral)s?\b", re.I)
_APOLOGY_PATTERN = re.compile(r"\b(?:apolog(?:y|ize|ise)|sorry|forgive|make\s+up)\b", re.I)
_FLORAL_DESIGN = re.compile(r"\b(?:floral|design|designs)\b", re.I)
_GIFT_VOUCHER_Q = re.compile(r"\bgift\s+voucher\b", re.I)
_CATALOG_INTENT = re.compile(
    r"\b(?:cakes?|flowers?|chocolates?|gifts?|hampers?|bouquets?|roses?|product|search|find|show|"
    r"deliver|track|VIMP)\b",
    re.I,
)


def _search_has_products(result: Any) -> bool:
    """Return True when a search_products MCP payload includes at least one hit."""
    if not isinstance(result, dict) or "error" in result:
        return False
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return False
    return bool(raw_results)


async def _broaden_budget_refinement_search(
    *,
    result: Any,
    enriched_args: dict[str, Any],
    state: AgentState,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> tuple[Any, dict[str, Any], list[ToolInvocation], bool]:
    """Retry empty budget searches via broaden ladder + focus fallback queries.

    Returns ``(result, args, tool_trace, broaden_applied)``.
    """
    if _search_has_products(result) or (isinstance(result, dict) and result.get("error")):
        return result, enriched_args, tool_trace, False

    working_trace = list(tool_trace)
    working_args = dict(enriched_args)
    broaden_applied = False
    intent_meta = state.get("intent_metadata")
    intent_dict = intent_meta if isinstance(intent_meta, dict) else None

    # Ladder: strip category / simplify q while keeping max_price.
    for _ in range(3):
        if _search_has_products(result):
            break
        broadened_args, _step = apply_first_broaden(
            working_args,
            preserve_max_price=True,
            intent_metadata=intent_dict,
        )
        if broadened_args is None:
            break
        if _is_duplicate_invocation(working_trace, SEARCH_PRODUCTS_TOOL, broadened_args):
            break
        broaden_applied = True
        _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
        injected = _inject_tool_currency(
            SEARCH_PRODUCTS_TOOL,
            broadened_args,
            state,
            currency,
        )
        broaden_result = await _invoke_tool_with_rate_limit_retry(
            SEARCH_PRODUCTS_TOOL,
            injected,
            kapruka_service=kapruka_service,
            client_ip=rate_limit_key,
            currency=currency,
        )
        broaden_result = _curate_search_trace_result(broaden_result, state=state)
        working_trace.append(
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": injected,
                "result": broaden_result,
            },
        )
        working_args = injected
        result = broaden_result
        if isinstance(result, dict) and result.get("error"):
            return result, working_args, working_trace, broaden_applied

    # Focus fallback qs (e.g. chocolate cake → chocolate) still under max_price.
    if not _search_has_products(result):
        max_price = working_args.get("max_price")
        for fallback_q in budget_focus_fallback_queries(dict(state)):
            fallback_args: dict[str, Any] = {
                "q": fallback_q,
                "currency": working_args.get("currency") or currency,
                "sort": "relevance",
            }
            if isinstance(max_price, (int, float)) and max_price > 0:
                fallback_args["max_price"] = float(max_price)
            if _is_duplicate_invocation(working_trace, SEARCH_PRODUCTS_TOOL, fallback_args):
                continue
            broaden_applied = True
            _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
            injected = _inject_tool_currency(
                SEARCH_PRODUCTS_TOOL,
                fallback_args,
                state,
                currency,
            )
            fallback_result = await _invoke_tool_with_rate_limit_retry(
                SEARCH_PRODUCTS_TOOL,
                injected,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
            fallback_result = _curate_search_trace_result(fallback_result, state=state)
            working_trace.append(
                {
                    "name": SEARCH_PRODUCTS_TOOL,
                    "args": injected,
                    "result": fallback_result,
                },
            )
            working_args = injected
            result = fallback_result
            if _search_has_products(result):
                break
            if isinstance(result, dict) and result.get("error"):
                break

    return result, working_args, working_trace, broaden_applied


def _persist_session_search_query(
    raw_q: str,
    *,
    intent_metadata: IntentMetadata | None = None,
) -> str:
    """Strip delivery cities before persisting session search topic."""
    stripped = strip_location_from_search_query(raw_q, intent_metadata)
    return stripped.strip() or raw_q.strip()


def _search_product_count(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return 0
    return len(raw_results)


_BUDGETED_GIFT_PHYSICAL_QUERIES: tuple[str, ...] = (
    "chocolates gift box",
    "gift hamper",
    "combo pack",
)


def _budgeted_gift_physical_queries(user_message: str) -> tuple[str, ...]:
    """Physical gift searches to run before voucher fallback on budgeted gift turns."""
    lowered = user_message.lower()
    queries: list[str] = []
    if re.search(r"\b(?:tea|teas)\b", lowered):
        queries.append("tea gift")
    if "hamper" in lowered:
        queries.append("gift hamper")
    if "chocolate" in lowered:
        queries.append("chocolates gift box")
    if re.search(r"\b(?:flower|flowers|rose|roses)\b", lowered):
        queries.append("fresh roses bouquet")
    if queries:
        return tuple(dict.fromkeys(queries))
    return _BUDGETED_GIFT_PHYSICAL_QUERIES


def _last_search_products_from_trace(
    tool_trace: list[ToolInvocation],
    *,
    state: AgentState | None = None,
) -> list[dict[str, Any]] | None:
    """Collect product dicts from the latest successful kapruka_search_products call."""
    for invocation in reversed(tool_trace):
        if invocation["name"] != SEARCH_PRODUCTS_TOOL:
            continue
        result = invocation["result"]
        if not _search_has_products(result):
            continue
        raw_results = result.get("results")
        if not isinstance(raw_results, list):
            continue
        products = [item for item in raw_results if isinstance(item, dict)]
        if not products:
            return None
        if state is None:
            return products
        user_message = _extract_latest_user_message(state.get("messages") or [])
        hybrid_context = state.get("hybrid_context") or {}
        return (
            apply_birthday_cake_curation(
                apply_puja_curation(
                    products,
                    query=user_message,
                    graph_context_available=has_graph_hybrid_context(hybrid_context),
                ),
                query=user_message,
                hybrid_context=hybrid_context,
                graph_context_available=has_graph_hybrid_context(hybrid_context),
                session_product_focus=state.get("session_product_focus"),
            )
            or None
        )
    return None


def _merge_search_results(
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> dict[str, Any]:
    """Merge two search_products payloads, deduping by product id (primary order first)."""
    if not isinstance(primary, dict):
        return secondary if isinstance(secondary, dict) else primary
    if not isinstance(secondary, dict):
        return primary
    if "error" in primary:
        return secondary if "error" not in secondary else primary
    if "error" in secondary:
        return primary

    seen: set[str] = set()
    merged_results: list[dict[str, Any]] = []

    for payload in (primary, secondary):
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            continue
        for product in raw_results:
            if not isinstance(product, dict):
                continue
            product_id = product.get("id")
            key = str(product_id) if product_id is not None else None
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged_results.append(product)

    merged = dict(primary)
    merged["results"] = merged_results
    return merged


def _dual_gift_physical_query(user_message: str) -> str:
    """Pick a physical-gift search query to pair with gift voucher searches."""
    lowered = user_message.lower()
    if "chocolate" in lowered:
        return "chocolates gift"
    if "hamper" in lowered:
        return "gift hamper"
    if "birthday" in lowered:
        return "birthday gift"
    return "gift hamper"


def _should_run_budgeted_gift_ideas_search(
    state: AgentState,
    user_message: str,
    *,
    already_ran: bool,
) -> bool:
    """True when the welcome chip or budgeted gift-ideas turn needs dual MCP search."""
    if already_ran:
        return False
    # Mid-chat budget refine must never pivot into gift-voucher discovery.
    focus = state.get("session_product_focus")
    focus_str = focus.strip() if isinstance(focus, str) else None
    if is_budget_refinement_message(user_message, session_product_focus=focus_str):
        return False
    if state.get("last_visible_products") or state.get("last_search_products"):
        return False
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    if not intent_metadata.get("budgeted_gift_discovery") and not is_budgeted_gift_ideas_message(
        user_message,
    ):
        return False
    budget_max = state.get("session_budget_max")
    if not isinstance(budget_max, (int, float)) or budget_max <= 0:
        return False
    session_q = state.get("session_search_query")
    return not (isinstance(session_q, str) and session_q.strip())


def _should_run_dual_gift_search(
    state: AgentState,
    tool_args: dict[str, Any],
    *,
    already_ran: bool,
) -> bool:
    """True when a budgeted gift turn should also search physical gifts."""
    if already_ran:
        return False
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    budget_max = intent_metadata.get("budget_max") or state.get("session_budget_max")
    if not isinstance(budget_max, (int, float)) or budget_max <= 0:
        return False
    query = str(tool_args.get("q") or tool_args.get("query") or "")
    if not _GIFT_VOUCHER_Q.search(query):
        return False
    user_message = _extract_latest_user_message(state.get("messages") or [])
    return bool(_GIFT_WORD.search(user_message))


def _should_run_situational_flowers_search(
    state: AgentState,
    user_message: str,
    *,
    already_ran: bool,
) -> bool:
    """True when a distress turn should auto-search apology/sympathy flowers."""
    if already_ran:
        return False
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    # Proactively show apology flowers even when the shopper did not name a product.
    return bool(intent_metadata.get("is_situational"))


def _situational_flowers_search_args(user_message: str) -> dict[str, Any]:
    """Build iteration-0 search args for situational flower/apology turns."""
    if _APOLOGY_PATTERN.search(user_message) or not _FLOWERS_REQUEST.search(user_message):
        query = "apology flowers"
    else:
        query = "roses bouquet"
    return {"q": query, "category": "Flowers"}


def _agent_tool_error_from_result(tool_name: str, result: dict[str, Any]) -> dict[str, str]:
    error_message = result.get("message")
    payload: dict[str, str] = {
        "tool": tool_name,
        "message": (
            str(error_message).strip()
            if isinstance(error_message, str) and error_message.strip()
            else str(result.get("error"))
        ),
    }
    error_code = result.get("error")
    if error_code is not None:
        code_str = str(error_code)
        if code_str == "429":
            code_str = "rate_limit_exceeded"
        payload["error"] = code_str
    retry_after = result.get("retry_after_seconds")
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        payload["retry_after_seconds"] = str(int(retry_after))
    elif isinstance(retry_after, str) and retry_after.isdigit():
        payload["retry_after_seconds"] = retry_after
    return payload


def _is_rate_limit_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    error_code = str(result.get("error") or "")
    return error_code in ("429", "rate_limit_exceeded")


def _rate_limit_retry_delay_seconds(result: dict[str, Any]) -> int:
    raw_retry = result.get("retry_after_seconds")
    if isinstance(raw_retry, (int, float)) and raw_retry > 0:
        return min(int(raw_retry), 5)
    if isinstance(raw_retry, str) and raw_retry.isdigit():
        return min(int(raw_retry), 5)
    return 1


async def _invoke_tool_with_rate_limit_retry(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    kapruka_service: KaprukaService,
    client_ip: str,
    currency: str | None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """Retry Kapruka MCP reads on rate limit with backoff and status SSE."""
    from lib.genai.completions import seconds_until_deadline

    if max_retries is None:
        remaining = seconds_until_deadline()
        # Under deadline pressure, skip 429 backoff so the turn can still finish.
        max_retries = 0 if remaining is not None and remaining < 30.0 else 2
    result: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        _emit_status(_status_message_for_tool(tool_name))
        result = await invoke_tool(
            tool_name,
            tool_args,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
            currency=currency or "LKR",
        )
        if not _is_rate_limit_result(result):
            return result
        if attempt >= max_retries:
            return result
        await asyncio.sleep(_rate_limit_retry_delay_seconds(result))
    return result


def _curate_search_trace_result(
    result: Any,
    *,
    state: AgentState,
) -> dict[str, Any] | Any:
    """Rewrite search trace payload so carousel/LLM see birthday-cake curated hits."""
    if not isinstance(result, dict) or not _search_has_products(result):
        return result
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return result
    products = [item for item in raw_results if isinstance(item, dict)]
    if not products:
        return result
    user_message = _extract_latest_user_message(state.get("messages") or [])
    hybrid_context = state.get("hybrid_context") or {}
    graph_up = has_graph_hybrid_context(hybrid_context)
    session_focus = state.get("session_product_focus")
    session_recipient = state.get("session_recipient_hint")
    session_occasion_val = state.get("session_occasion")
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    session_flavor_hint = state.get("session_flavor_hint")
    if not isinstance(session_flavor_hint, str) or not session_flavor_hint.strip():
        meta_flavor = (
            intent_metadata.get("session_flavor_hint")
            if isinstance(
                intent_metadata,
                dict,
            )
            else None
        )
        session_flavor_hint = (
            meta_flavor.strip() if isinstance(meta_flavor, str) and meta_flavor.strip() else None
        )
    topic_pivot = bool(
        isinstance(intent_metadata, dict) and intent_metadata.get("topic_pivot"),
    )
    strict_budget = has_explicit_budget_constraint(
        user_message,
        state.get("session_budget_max")
        if isinstance(state.get("session_budget_max"), (int, float))
        else None,
        topic_pivot=topic_pivot,
    )
    session_budget = state.get("session_budget_max")
    if (
        not strict_budget
        and is_flower_fruit_intent(user_message)
        and isinstance(session_budget, (int, float))
        and session_budget > 0
    ):
        strict_budget = True
    curated = apply_anniversary_curation(
        apply_gift_curation(
            apply_birthday_cake_curation(
                apply_puja_curation(
                    products,
                    query=user_message,
                    graph_context_available=graph_up,
                ),
                query=user_message,
                hybrid_context=hybrid_context,
                graph_context_available=graph_up,
                session_product_focus=session_focus,
                session_flavor_hint=session_flavor_hint,
            ),
            session_product_focus=session_focus if isinstance(session_focus, str) else None,
            user_message=user_message,
            hybrid_context=hybrid_context,
        ),
        query=user_message,
        hybrid_context=hybrid_context,
        session_occasion=session_occasion_val if isinstance(session_occasion_val, str) else None,
    )
    curated = apply_recipient_curation(
        curated,
        session_recipient if isinstance(session_recipient, str) else None,
    )
    curated = demote_non_chocolate_for_chocolate_focus(
        curated,
        user_message,
        session_product_focus=session_focus if isinstance(session_focus, str) else None,
    )
    curated = demote_non_tea_for_tea_focus(
        curated,
        user_message,
        session_product_focus=session_focus if isinstance(session_focus, str) else None,
    )
    curated = filter_excluded_category_hints(
        curated,
        hybrid_context,
        session_product_focus=session_focus if isinstance(session_focus, str) else None,
        query=user_message,
    )
    curated = demote_off_focus_products(
        curated,
        session_focus if isinstance(session_focus, str) else None,
    )
    curated = filter_gift_noise_products(curated, strict=strict_budget)
    budget_max = state.get("session_budget_max")
    currency = state.get("currency") or state.get("session_budget_currency") or "LKR"
    if not topic_pivot and isinstance(budget_max, (int, float)) and budget_max > 0:
        from lib.chat.product_curation import sort_and_filter_by_budget

        curated = sort_and_filter_by_budget(
            curated,
            float(budget_max),
            str(currency),
            strict_in_budget=strict_budget,
        )
    if (
        strict_budget
        and is_flower_fruit_intent(user_message)
        and isinstance(budget_max, (int, float))
        and budget_max > 0
    ):
        from lib.chat.product_curation import ensure_flower_price_tier_diversity

        curated = ensure_flower_price_tier_diversity(curated, float(budget_max))
    # Budget-refine turns must not wipe in-budget MCP hits via aggressive curation.
    # Keep gift-noise filters; only recover focus-compatible in-budget gifts.
    if (
        not curated
        and products
        and is_budget_refinement_message(user_message)
        and isinstance(budget_max, (int, float))
        and budget_max > 0
    ):
        in_budget = [
            item
            for item in products
            if (price := product_price_amount(item)) is not None and price <= float(budget_max)
        ]
        cleaned = filter_gift_noise_products(in_budget, strict=True)
        if cleaned:
            updated = dict(result)
            updated["results"] = cleaned
            return updated
    if curated == products:
        return result
    updated = dict(result)
    updated["results"] = curated
    return updated


def _turn_needs_catalog(state: AgentState) -> bool:
    """Return True when the user turn likely needs Kapruka catalog tool calls."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if _CATALOG_INTENT.search(user_message):
        return True
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    return bool(
        intent_metadata.get("requires_delivery_validation") or intent_metadata.get("target_city"),
    )


def _graph_hint_iteration_limit(state: AgentState) -> int | None:
    """Cap planner iterations when graph hints are strong and retrieval is healthy."""
    intent_metadata = state.get("intent_metadata") or {}
    if intent_metadata.get("graph_degraded"):
        return MAX_ITERATIONS
    hybrid_context = state.get("hybrid_context") or {}
    if has_strong_hybrid_hints(hybrid_context):
        return GRAPH_HINTS_MAX_ITERATIONS
    return None


def _max_iterations_for_state(state: AgentState, refined_intent: Intent | None) -> int:
    """Cap planner iterations for fast utility/general turns that need no catalog."""
    graph_hint_cap = _graph_hint_iteration_limit(state)
    if graph_hint_cap is not None:
        return graph_hint_cap
    if refined_intent == "discovery":
        user_message = _extract_latest_user_message(state.get("messages") or [])
        if is_confident_discovery_turn(
            user_message,
            state.get("hybrid_context") or {},
            currency=_resolve_currency(state),
            intent_metadata=state.get("intent_metadata"),
            state=dict(state),
        ):
            return CONFIDENT_DISCOVERY_MAX_ITERATIONS
    if refined_intent != "general":
        return MAX_ITERATIONS
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    if intent_metadata.get("is_situational"):
        return MAX_ITERATIONS
    if _turn_needs_catalog(state):
        return MAX_ITERATIONS
    return UTILITY_GENERAL_MAX_ITERATIONS


def _initial_iteration_limit(state: AgentState) -> int:
    """Pre-loop cap when hybrid hints make a single planner pass sufficient."""
    graph_hint_cap = _graph_hint_iteration_limit(state)
    if graph_hint_cap is not None:
        return graph_hint_cap
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if is_confident_discovery_turn(
        user_message,
        state.get("hybrid_context") or {},
        currency=_resolve_currency(state),
        intent_metadata=state.get("intent_metadata"),
        state=dict(state),
    ):
        return CONFIDENT_DISCOVERY_MAX_ITERATIONS
    return MAX_ITERATIONS


def _search_query_from_result(result: Any) -> str | None:
    """Extract applied_filters.q from a kapruka_search_products payload."""
    if not isinstance(result, dict):
        return None
    filters = result.get("applied_filters")
    if isinstance(filters, dict):
        query = filters.get("q")
        if isinstance(query, str) and query.strip():
            return query.strip()
    return None


def _accessory_ratio(products: list[dict[str, Any]]) -> float:
    if not products:
        return 0.0
    accessories = sum(1 for product in products if is_cake_accessory(product))
    return accessories / len(products)


def _trace_has_dated_check_delivery(tool_trace: list[ToolInvocation]) -> bool:
    for invocation in reversed(tool_trace):
        if invocation["name"] != CHECK_DELIVERY_TOOL:
            continue
        args = invocation.get("args")
        if isinstance(args, dict):
            delivery_date = args.get("delivery_date")
            if isinstance(delivery_date, str) and delivery_date.strip():
                return True
    return False


def _resolve_delivery_product_id(state: AgentState) -> str | None:
    """Pick a catalog product id for perishable-aware delivery checks."""
    for key in ("last_visible_products", "last_search_products"):
        products = state.get(key)
        if not isinstance(products, list) or not products:
            continue
        first = products[0]
        if isinstance(first, dict):
            product_id = first.get("id")
            if isinstance(product_id, str) and product_id.strip():
                return product_id.strip()
    return None


def _delivery_check_pending(
    state: AgentState,
    tool_trace: list[ToolInvocation],
    user_message: str,
) -> bool:
    """True when city+date are known but kapruka_check_delivery has not run with a date."""
    if not is_delivery_context_relevant_turn(dict(state), user_message):
        return False
    session_date = state.get("session_delivery_date") or state.get("delivery_date")
    if not (isinstance(session_date, str) and session_date.strip()):
        return False
    canonical_city = state.get("delivery_city_canonical") or state.get(
        "session_delivery_city_canonical",
    )
    if not (isinstance(canonical_city, str) and canonical_city.strip()):
        return False
    if _trace_has_dated_check_delivery(tool_trace):
        return False
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    if intent_metadata.get("requires_delivery_validation") or intent_metadata.get("target_city"):
        return True
    if bool(state.get("session_delivery_city_confirmed")):
        return True
    session_focus = state.get("session_product_focus")
    if isinstance(session_focus, str) and session_focus.strip().lower() in {
        "flowers",
        "cake",
        "gift",
        "chocolate",
    }:
        return True
    return _has_perishable_gift_intent(user_message) or is_flower_fruit_intent(user_message)


def _should_run_discovery_city_gift_search(
    state: AgentState,
    user_message: str,
    *,
    already_ran: bool,
) -> bool:
    """Search catalog first when a gift turn names a city but no delivery date."""
    if already_ran:
        return False
    return should_defer_delivery_date(state, user_message)


def _discovery_city_gift_search_args(state: AgentState, user_message: str) -> dict[str, Any]:
    """Build iteration-0 search args for city-scoped gift discovery."""
    enriched_args = merge_planner_search_args(
        {"q": "birthday cake"},
        user_message=user_message,
        hybrid_context=state.get("hybrid_context") or {},
        currency=_resolve_currency(state),
        intent_metadata=state.get("intent_metadata"),
        state=dict(state),
    )
    if is_broad_cakes_query(user_message) or is_birthday_cake_intent(user_message):
        enriched_args["q"] = "birthday cake"
        enriched_args.setdefault("category", "Birthday")
    elif is_flower_fruit_intent(user_message):
        enriched_args.setdefault("q", "roses bouquet")
        enriched_args.setdefault("category", "Flowers")
    return enriched_args


def _should_force_finish_after_search(
    state: AgentState,
    tool_trace: list[ToolInvocation],
) -> bool:
    """Return True when a successful search should end the loop on the next iteration."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if should_defer_delivery_date(state, user_message):
        return True
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    pending_delivery = intent_metadata.get("requires_delivery_validation") or bool(
        intent_metadata.get("target_city")
    )
    if pending_delivery:
        already_checked = any(
            invocation["name"] == CHECK_DELIVERY_TOOL for invocation in tool_trace
        )
        if not already_checked:
            return False
    return True


def _format_planner_query_rewrite_hints(
    user_message: str,
    *,
    message_count: int = 1,
    budget_max: float | None = None,
    graph_context_available: bool = False,
    graph_degraded: bool = False,
    session_product_focus: str | None = None,
    session_occasion: str | None = None,
    topic_pivot: bool = False,
) -> str:
    """Soft search-query rewrite suggestions for broad cake and mom/birthday turns."""
    hints: list[str] = []
    if graph_degraded:
        hints.append(
            "GraphRAG returned no products this turn — rely on kapruka_search_products "
            "with broad discovery queries rather than graph-based context."
        )
    has_budget = budget_max is not None and budget_max > 0
    bare_focus = is_bare_category_pivot(user_message)
    if bare_focus:
        flower_q = (
            "fresh flowers"
            if bare_focus == "flowers" and "fresh" in user_message.lower()
            else bare_focus
        )
        hints.append(
            "Topic pivot with bare category: prefer action call_tool with "
            f'kapruka_search_products using literal q="{flower_q}" '
            '(or q="fresh flowers" when the customer said flowers) — do not inherit '
            "prior occasion context. Do NOT ask_user for occasion before searching. "
            "Only ask_user if search returns no products."
        )
    elif topic_pivot and re.search(
        r"\b(?:cakes?|flowers?|roses?|chocolates?|bouquets?)\b",
        user_message,
        re.I,
    ):
        hints.append(
            "Topic pivot naming a product category: prefer kapruka_search_products "
            "immediately — do not ask_user for occasion, recipient, or budget first."
        )
    if session_product_focus == "cake" and _FLORAL_DESIGN.search(user_message):
        hints.append(
            "Session shopping focus is cake: prefer kapruka_search_products with "
            'q="floral birthday cake" and category="Birthday" rather than jewelry or apparel.'
        )
    if message_count > 1 and _SHORT_CATEGORY_REPLY.match(user_message.strip()) and not topic_pivot:
        hints.append(
            "Follow-up category reply after a prior clarifying turn: prefer action call_tool "
            'with kapruka_search_products (e.g. "cakes" → q="birthday cake"; '
            '"flowers" → q="fresh roses bouquet") rather than ask_user.'
        )
    if (
        _CAKES_BROAD.search(user_message)
        and not _BIRTHDAY_CAKE.search(user_message)
        and not topic_pivot
    ):
        hints.append(
            'Broad "cakes" query: prefer kapruka_search_products with q="birthday cake" '
            "unless the customer named a specific cake type."
        )
    if _BIRTHDAY_CAKE.search(user_message) or is_birthday_cake_intent(user_message):
        birthday_hint = (
            'Explicit birthday cake request: prefer kapruka_search_products with q="birthday cake" '
            'and category="Birthday"; avoid generic chocolate or dessert-only searches that omit '
            "cake products."
        )
        if graph_context_available:
            birthday_hint += " Graph exclude_categories lists dessert departments to deprioritize."
        hints.append(birthday_hint)
    if _MOM_BIRTHDAY.search(user_message):
        hints.append(
            "Mom/mother + birthday occasion: bias search q toward birthday cakes, flowers, "
            "or combopack/combo gifts unless the customer specified another product type."
        )
    if _BROAD_GIFTS.match(user_message.strip()):
        if has_budget:
            hints.append(
                f'Budgeted "gifts" query: prefer two kapruka_search_products calls — '
                f'(1) q="gift voucher" max_price={budget_max}, then '
                f'(2) q="gift hamper" max_price={budget_max} — rather than ask_user.'
            )
        else:
            hints.append(
                'Vague "gifts" query with no occasion, recipient, or budget: prefer action '
                "ask_user before kapruka_search_products."
            )
    elif has_budget and _GIFT_WORD.search(user_message):
        hints.append(
            f"Budgeted gift query: prefer two kapruka_search_products calls — "
            f'(1) q="gift voucher" max_price={budget_max}, then '
            f'(2) q="gift hamper" max_price={budget_max} — before ask_user.'
        )
    if _FLOWERS_REQUEST.search(user_message):
        color_match = re.search(
            r"\b(blush|pink|red|white|yellow|purple|lavender)\b",
            user_message,
            re.I,
        )
        if color_match:
            color = color_match.group(1).lower()
            flower = "roses" if re.search(r"\broses?\b", user_message, re.I) else "flowers"
            hints.append(
                f"Color+flower request: prefer kapruka_search_products with literal "
                f'q="{color} {flower}" and category="Flowers" — do not rewrite to generic '
                "fresh cut roses. If top results lack the color modifier, retry once with "
                "the same literal q before finish."
            )
        else:
            hints.append(
                "Flowers/roses/bouquet request: prefer kapruka_search_products q emphasizing "
                "fresh cut roses or bouquets. If results are only silk, artificial, soap, or "
                "paper florals, try one broader fresh-flowers search before finish."
            )
    if is_flower_fruit_intent(user_message):
        puja_avoid = (
            "Flower/fruit request: avoid puja, pooja, watti, and religious offering products; "
            "prefer fresh flowers, fruit baskets, and bouquets."
        )
        if graph_context_available:
            puja_avoid += " Graph exclude_categories hint lists puja/religious categories to skip."
        hints.append(puja_avoid)
    _is_anniversary = re.search(r"\banniversary\b", user_message, re.I) or (
        isinstance(session_occasion, str) and "anniversary" in session_occasion.lower()
    )
    if _is_anniversary:
        ann_hint = (
            'Anniversary occasion: prefer kapruka_search_products with q="anniversary flowers" '
            'or q="anniversary gift hamper"; avoid greeting cards, watch boxes, storage boxes, '
            'and gift vouchers. Do not set category="Flowers" — anniversary flower gifts '
            "include cake+rose combos catalogued under Cakes."
        )
        if graph_context_available:
            ann_hint += " Graph exclude_categories lists cards/vouchers to skip."
        hints.append(ann_hint)
    if not hints:
        return ""
    bullet_lines = "\n".join(f"- {hint}" for hint in hints)
    return f"Query rewrite hints (soft — explicit customer wording wins):\n{bullet_lines}"


def _format_hybrid_soft_hints(state: AgentState) -> str:
    """Serialize hybrid_context hints and preferences as soft narrative context."""
    hybrid_context = state.get("hybrid_context") or {}
    hints = hybrid_context.get("hints") or {}
    preferences = hybrid_context.get("preferences") or {}
    if not hints and not preferences:
        return ""
    payload = {"hints": hints, "preferences": preferences}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_planner_system_instruction(
    state: AgentState,
    *,
    tool_trace: list[ToolInvocation],
) -> str:
    """Compose planner system instruction with memory and hybrid soft hints."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    delivery_relevant = is_delivery_context_relevant_turn(dict(state), user_message)
    instruction = (
        f"{PLANNER_SYSTEM_INSTRUCTION}\n\n"
        f"Today in Sri Lanka: {colombo_today_iso()}\n"
        "For kapruka_check_delivery, delivery_date must be YYYY-MM-DD on or after today."
    )
    currency = state.get("currency")
    if isinstance(currency, str) and currency.strip():
        instruction += (
            f"\n\nDisplay currency (authoritative): {currency.strip()} — use for all price "
            "filters; do not ask the customer to choose currency."
        )
    if delivery_relevant:
        session_date = state.get("session_delivery_date") or state.get("delivery_date")
        if isinstance(session_date, str) and session_date.strip():
            instruction += (
                f"\n\nResolved delivery date: {session_date.strip()} — do not ask_user for the "
                "date when delivery validation is needed; use this date in kapruka_check_delivery."
            )
        canonical_city = state.get("session_delivery_city_canonical") or state.get(
            "delivery_city_canonical",
        )
        if isinstance(canonical_city, str) and canonical_city.strip():
            instruction += f"\n\nResolved delivery city: {canonical_city.strip()}."
    session_focus = state.get("session_product_focus")
    if isinstance(session_focus, str) and session_focus.strip():
        instruction += f"\n\nSession shopping focus: {session_focus.strip()} (from earlier turn)."
    session_search_q = state.get("session_search_query")
    if isinstance(session_search_q, str) and session_search_q.strip():
        instruction += (
            f"\n\nSession search topic: {session_search_q.strip()} — reuse for budget refinements."
        )
    zep_memory_facts = state.get("zep_memory_facts")
    if zep_memory_facts:
        scoped_facts = scope_memory_facts_for_turn(zep_memory_facts, user_message)
        if scoped_facts:
            facts_block = format_memory_facts_block(scoped_facts).replace(
                "Prior session facts (context only):",
                "Prior context — do not mention unless the customer asks:",
            )
            instruction += facts_block
    hybrid_block = _format_hybrid_soft_hints(state)
    if hybrid_block:
        instruction += (
            "\n\nHybrid context (soft hints only — defer to explicit user message):\n"
            f"{hybrid_block}"
        )
    prior = format_planner_prior_iterations(tool_trace)
    if prior:
        instruction += f"\n\nPrior tool iterations (summarized):\n{prior}"
    return instruction


def _build_planner_user_prompt(state: AgentState) -> str:
    """User turn content for the planner — latest message only."""
    messages = state.get("messages") or []
    user_message = _extract_latest_user_message(messages)
    prompt = f"Customer message:\n{user_message}"
    currency = state.get("currency")
    if isinstance(currency, str) and currency.strip():
        prompt += f"\n\nSession display currency: {currency.strip()}"
    if is_delivery_context_relevant_turn(dict(state), user_message):
        session_date = state.get("session_delivery_date") or state.get("delivery_date")
        if isinstance(session_date, str) and session_date.strip():
            prompt += f"\n\nResolved delivery date: {session_date.strip()}"
    session_focus = state.get("session_product_focus")
    if isinstance(session_focus, str) and session_focus.strip():
        prompt += f"\n\nSession shopping focus: {session_focus.strip()}"
    session_recipient = state.get("session_recipient_hint")
    if isinstance(session_recipient, str) and session_recipient.strip():
        prompt += f"\n\nSession recipient: {session_recipient.strip()}"
    session_flavor = state.get("session_flavor_hint")
    if isinstance(session_flavor, str) and session_flavor.strip():
        prompt += f"\n\nSession flavor/style hint: {session_flavor.strip()}"
    session_search_q = state.get("session_search_query")
    if isinstance(session_search_q, str) and session_search_q.strip():
        prompt += f"\n\nSession search topic: {session_search_q.strip()}"
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    target_city = intent_metadata.get("target_city")
    if isinstance(target_city, str) and target_city.strip():
        prompt += f"\n\nTarget delivery city: {target_city.strip()}"
    budget_max = intent_metadata.get("budget_max")
    hybrid_context = state.get("hybrid_context") or {}
    session_occasion = state.get("session_occasion")
    rewrite_hints = _format_planner_query_rewrite_hints(
        user_message,
        message_count=len(messages),
        budget_max=budget_max if isinstance(budget_max, (int, float)) else None,
        graph_context_available=has_graph_hybrid_context(hybrid_context),
        graph_degraded=bool(intent_metadata.get("graph_degraded")),
        session_product_focus=session_focus if isinstance(session_focus, str) else None,
        session_occasion=session_occasion if isinstance(session_occasion, str) else None,
        topic_pivot=bool(intent_metadata.get("topic_pivot")),
    )
    if rewrite_hints:
        prompt += f"\n\n{rewrite_hints}"
    return prompt


async def _plan_next_step(
    *,
    state: AgentState,
    tool_trace: list[ToolInvocation],
) -> AgentPlannerStep:
    """NVIDIA NIM planner call with structured output."""
    try:
        step = await generate_content(
            model=PLANNER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _build_planner_system_instruction(state, tool_trace=tool_trace),
                },
                {"role": "user", "content": _build_planner_user_prompt(state)},
            ],
            response_schema=AgentPlannerStep,
            temperature=0,
        )
        if not isinstance(step, AgentPlannerStep):
            msg = "NVIDIA NIM returned unparseable planner step"
            raise ValueError(msg)
        return step
    except Exception as exc:
        # Preserve transient NIM errors so the loop can degrade gracefully instead of
        # hard-failing the turn (wrapping as ValueError would strip the type and
        # trip the generic "Something went wrong" SSE error path).
        if is_transient_nim_error(exc):
            raise
        msg = f"NVIDIA NIM planner call failed: {exc}"
        raise ValueError(msg) from exc


def _retry_after_seconds(exc: BaseException) -> int | None:
    """Best-effort Retry-After (seconds) from a NIM rate-limit response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if isinstance(raw, str) and raw.strip().isdigit():
        return max(1, int(raw.strip()))
    return None


def _planner_transient_error(exc: BaseException) -> dict[str, str]:
    """Build an agent_tool_error dict so generate_response degrades gracefully.

    A rate-limited or timed-out planner call would otherwise escape agent_loop
    uncaught and be rendered as the generic "Something went wrong" SSE error.
    Routing it through the tool_error path lets generate_response show friendly
    copy, reuse any prior carousel products, and attach the retry banner.
    """
    if is_rate_limited(exc):
        error: dict[str, str] = {
            "tool": SEARCH_PRODUCTS_TOOL,
            "error": "rate_limit_exceeded",
            "message": "NVIDIA NIM is temporarily rate limited.",
        }
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            error["retry_after_seconds"] = str(retry_after)
        return error
    return {
        "tool": SEARCH_PRODUCTS_TOOL,
        "error": "timeout",
        "message": (
            "That took too long — please try again. "
            "Your last results are still above if you want to refine or pick another gift."
        ),
    }


def _planner_rate_limit_error(exc: BaseException) -> dict[str, str]:
    """Backward-compatible alias for rate-limit planner degradation."""
    return _planner_transient_error(exc)


def _emit_status(message: str) -> None:
    """Emit a LangGraph custom stream status event when a writer is available."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    if writer is not None:
        writer({"type": "status", "message": message})


def _emit_provisional_carousel(
    products: list[dict[str, Any]],
    *,
    state: AgentState,
    user_message: str,
) -> None:
    """Stream a provisional product carousel before generate_response finishes."""
    if not products:
        return
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    if writer is None:
        return
    try:
        from graphs.nodes.generate_response import (
            build_products_carousel_html,
            render_carousel_oob_html,
        )
    except Exception:  # noqa: BLE001 — never fail the turn on provisional UI
        return
    budget_max = state.get("session_budget_max")
    currency = str(state.get("currency") or "LKR")
    products_html = build_products_carousel_html(
        {"kapruka_search_products": {"results": products}},
        budget_max=float(budget_max) if isinstance(budget_max, (int, float)) else None,
        currency=currency,
        user_message=user_message,
        session_product_focus=state.get("session_product_focus")
        if isinstance(state.get("session_product_focus"), str)
        else None,
        last_search_products=products,
        visible_products=products,
        allow_stale_fallback=False,
    )
    if not products_html:
        return
    # Fixed provisional slot so OOB can update; streaming also appends if missing.
    slot_id = "carousel-slot-provisional"
    carousel_oob = render_carousel_oob_html(products_html, carousel_slot_id=slot_id)
    writer({"type": "carousel", "html": carousel_oob, "slot_id": slot_id})


def _status_message_for_tool(tool_name: str) -> str:
    return _TOOL_STATUS_MESSAGES.get(tool_name, _DEFAULT_STATUS_MESSAGE)


def _args_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_duplicate_invocation(
    tool_trace: list[ToolInvocation],
    tool_name: str,
    tool_args: dict[str, Any],
) -> bool:
    """Return True when the same tool+args already appear in the trace."""
    candidate = canonical_tool_args_for_dedup(tool_name, tool_args)
    for invocation in tool_trace:
        prior = canonical_tool_args_for_dedup(invocation["name"], invocation["args"])
        if invocation["name"] == tool_name and _args_equal(prior, candidate):
            return True
    return False


def _trace_has_check_delivery(tool_trace: list[ToolInvocation]) -> bool:
    return any(invocation["name"] == CHECK_DELIVERY_TOOL for invocation in tool_trace)


def _fast_path_agent_loop_updates(
    tool_trace: list[ToolInvocation],
    *,
    tool_call_count: int,
    refined_intent: Intent | None = "general",
    exit_reason: str = "finish",
    session_delivery_date_update: str | None = None,
    session_search_query_update: str | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "tool_trace": tool_trace,
        "tool_call_count": tool_call_count,
        "agent_loop_done": True,
        "agent_loop_exit_reason": exit_reason,
        "agent_loop_iterations": 0,
        "intent": refined_intent,
    }
    if tool_trace:
        updates["tool_results"] = {
            invocation["name"]: invocation["result"] for invocation in tool_trace
        }
    if session_delivery_date_update is not None:
        updates["session_delivery_date"] = session_delivery_date_update
        updates["delivery_date"] = session_delivery_date_update
    if session_search_query_update is not None:
        updates["session_search_query"] = session_search_query_update
    return updates


async def _run_pending_delivery_check(
    state: AgentState,
    tool_trace: list[ToolInvocation],
    *,
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
    user_message: str,
) -> tuple[list[ToolInvocation], int, str | None]:
    """Run kapruka_check_delivery when city+date are known but not yet in the trace."""
    tool_call_count = 0
    session_delivery_date_update: str | None = None
    if _trace_has_check_delivery(tool_trace):
        return tool_trace, tool_call_count, session_delivery_date_update
    if not _delivery_check_pending(state, tool_trace, user_message):
        return tool_trace, tool_call_count, session_delivery_date_update

    canonical_city = state.get("delivery_city_canonical") or state.get(
        "session_delivery_city_canonical",
    )
    session_date = state.get("session_delivery_date") or state.get("delivery_date")
    if not (
        isinstance(canonical_city, str)
        and canonical_city.strip()
        and isinstance(session_date, str)
        and session_date.strip()
    ):
        return tool_trace, tool_call_count, session_delivery_date_update

    delivery_args: dict[str, Any] = {
        "city": canonical_city.strip(),
        "delivery_date": session_date.strip(),
    }
    product_id = _resolve_delivery_product_id(state)
    if product_id:
        delivery_args["product_id"] = product_id
    delivery_args = _inject_tool_currency(
        CHECK_DELIVERY_TOOL,
        delivery_args,
        state,
        currency,
    )
    if _is_duplicate_invocation(tool_trace, CHECK_DELIVERY_TOOL, delivery_args):
        return tool_trace, tool_call_count, session_delivery_date_update

    _emit_status(_status_message_for_tool(CHECK_DELIVERY_TOOL))
    delivery_result = await invoke_tool(
        CHECK_DELIVERY_TOOL,
        delivery_args,
        kapruka_service=kapruka_service,
        client_ip=rate_limit_key,
        currency=currency,
    )
    tool_trace = [
        *tool_trace,
        {"name": CHECK_DELIVERY_TOOL, "args": delivery_args, "result": delivery_result},
    ]
    tool_call_count = 1
    if isinstance(delivery_result, dict) and not delivery_result.get("error"):
        session_delivery_date_update = delivery_args["delivery_date"]
    return tool_trace, tool_call_count, session_delivery_date_update


def _confident_discovery_fast_path_blocked(
    state: AgentState,
    user_message: str,
) -> bool:
    """True when another deterministic path should run instead of confident discovery."""
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    if intent_metadata.get("is_situational"):
        return True
    if _should_run_budgeted_gift_ideas_search(state, user_message, already_ran=False):
        return True
    if is_budget_refinement_message(user_message) and (
        state.get("session_search_query")
        or state.get("session_product_focus")
        or state.get("last_search_products")
    ):
        return True
    if _should_run_situational_flowers_search(state, user_message, already_ran=False):
        return True
    return bool(is_product_detail_turn(user_message))


async def _retry_anniversary_search_if_empty(
    result: dict[str, Any],
    enriched_args: dict[str, Any],
    *,
    state: AgentState,
    user_message: str,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> tuple[dict[str, Any], dict[str, Any], list[ToolInvocation], int]:
    """Run parallel fallback anniversary searches when the primary query is empty."""
    hybrid_context = state.get("hybrid_context") or {}
    if _search_has_products(result) or not is_anniversary_occasion_intent(
        user_message,
        hybrid_context,
    ):
        return result, enriched_args, tool_trace, 0

    current_q = str(enriched_args.get("q") or "").strip().lower()
    fallback_queries = [
        query
        for query in anniversary_fallback_search_queries(user_message)
        if query.strip().lower() != current_q
    ][:BUDGETED_GIFT_SEARCH_CONCURRENCY]
    if not fallback_queries:
        return result, enriched_args, tool_trace, 0

    search_sem = asyncio.Semaphore(BUDGETED_GIFT_SEARCH_CONCURRENCY)
    extra_trace = list(tool_trace)
    extra_calls = 0

    async def _run_retry(query: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        retry_args = _inject_tool_currency(
            SEARCH_PRODUCTS_TOOL,
            {**enriched_args, "q": query},
            state,
            currency,
        )
        if _is_duplicate_invocation(extra_trace, SEARCH_PRODUCTS_TOOL, retry_args):
            return retry_args, None
        async with search_sem:
            retry_result = await _invoke_tool_with_rate_limit_retry(
                SEARCH_PRODUCTS_TOOL,
                retry_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
        retry_result = _curate_search_trace_result(retry_result, state=state)
        return retry_args, retry_result

    _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
    retry_pairs = await asyncio.gather(
        *[_run_retry(query) for query in fallback_queries],
        return_exceptions=True,
    )
    for pair in retry_pairs:
        if isinstance(pair, BaseException):
            continue
        retry_args, retry_result = pair
        if retry_result is None:
            continue
        extra_trace.append(
            _tool_trace_entry(SEARCH_PRODUCTS_TOOL, retry_args, retry_result),
        )
        extra_calls += 1
        if _search_has_products(retry_result):
            return retry_result, retry_args, extra_trace, extra_calls
    return result, enriched_args, extra_trace, extra_calls


def _literal_color_flower_query(user_message: str) -> str | None:
    """Return literal 'blush roses'-style q when the customer named color+flower."""
    match = re.search(
        r"\b(blush|pink|red|white|yellow|purple|lavender)\s+"
        r"(roses?|flowers?|bouquets?)\b",
        user_message,
        re.I,
    )
    if not match:
        return None
    color = match.group(1).lower()
    flower = match.group(2).lower()
    if flower.startswith("rose"):
        flower = "roses"
    elif flower.startswith("flower"):
        flower = "flowers"
    else:
        flower = "bouquet"
    return f"{color} {flower}"


def _top_results_lack_color_modifier(
    result: dict[str, Any],
    *,
    color: str,
    limit: int = 5,
) -> bool:
    """True when none of the top search hits mention the requested color."""
    raw = result.get("results")
    if not isinstance(raw, list) or not raw:
        return True
    color_re = re.compile(rf"\b{re.escape(color)}\b", re.I)
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        blob = f"{item.get('name') or ''} {item.get('summary') or ''}"
        if color_re.search(blob):
            return False
    return True


async def _maybe_retry_literal_color_flower_search(
    result: dict[str, Any],
    enriched_args: dict[str, Any],
    *,
    state: AgentState,
    user_message: str,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> tuple[dict[str, Any], dict[str, Any], list[ToolInvocation], int]:
    """One-shot retry with literal color+flower q when top hits miss the color."""
    literal_q = _literal_color_flower_query(user_message)
    if not literal_q:
        return result, enriched_args, tool_trace, 0
    color = literal_q.split()[0]
    if not _top_results_lack_color_modifier(result, color=color):
        return result, enriched_args, tool_trace, 0
    current_q = str(enriched_args.get("q") or "").strip().lower()
    if current_q == literal_q and _search_has_products(result):
        # Already searched literally — still weak; nothing more to do.
        return result, enriched_args, tool_trace, 0
    retry_args = _inject_tool_currency(
        SEARCH_PRODUCTS_TOOL,
        {
            **enriched_args,
            "q": literal_q,
            "category": "Flowers",
            "sort": "relevance",
        },
        state,
        currency,
    )
    if _is_duplicate_invocation(tool_trace, SEARCH_PRODUCTS_TOOL, retry_args):
        return result, enriched_args, tool_trace, 0
    _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
    retry_result = await _invoke_tool_with_rate_limit_retry(
        SEARCH_PRODUCTS_TOOL,
        retry_args,
        kapruka_service=kapruka_service,
        client_ip=rate_limit_key,
        currency=currency,
        max_retries=0,
    )
    retry_result = _curate_search_trace_result(retry_result, state=state)
    updated_trace = [
        *tool_trace,
        _tool_trace_entry(SEARCH_PRODUCTS_TOOL, retry_args, retry_result),
    ]
    if _search_has_products(retry_result):
        return retry_result, retry_args, updated_trace, 1
    return result, enriched_args, updated_trace, 1


def _literal_named_product_query(user_message: str) -> str | None:
    """Return a literal blush/combo flower query when the customer named one."""
    stripped = user_message.strip()
    if not stripped:
        return None
    if not re.search(r"\b(?:blush|combo|combopack)\b", stripped, re.I):
        return None
    color_flower = re.search(
        r"\b(blush|pink|red|white|yellow|purple|lavender)\s+"
        r"(roses?|flowers?|bouquets?)\b",
        stripped,
        re.I,
    )
    if color_flower:
        color = color_flower.group(1).lower()
        flower = color_flower.group(2).lower()
        if flower.startswith("rose"):
            flower = "roses"
        elif flower.startswith("flower"):
            flower = "flowers"
        else:
            flower = "bouquet"
        literal_q = f"{color} {flower}"
        if re.search(r"\bcombo\b", stripped, re.I):
            literal_q = f"{literal_q} combo"
        return literal_q
    if re.search(r"\bcombo\b", stripped, re.I) and re.search(r"\broses?\b", stripped, re.I):
        return "roses combo"
    return None


def _top_results_lack_named_tokens(
    result: dict[str, Any],
    *,
    tokens: tuple[str, ...],
    limit: int = 5,
) -> bool:
    """True when none of the top search hits mention any of the distinctive tokens."""
    raw = result.get("results")
    if not isinstance(raw, list) or not raw:
        return True
    token_re = re.compile(
        r"\b(?:" + "|".join(re.escape(token) for token in tokens) + r")\b",
        re.I,
    )
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        blob = f"{item.get('name') or ''} {item.get('summary') or ''}"
        if token_re.search(blob):
            return False
    return True


async def _maybe_retry_literal_named_product_search(
    result: dict[str, Any],
    enriched_args: dict[str, Any],
    *,
    state: AgentState,
    user_message: str,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> tuple[dict[str, Any], dict[str, Any], list[ToolInvocation], int]:
    """One-shot retry when blush/combo query returns generic rose hits."""
    literal_q = _literal_named_product_query(user_message)
    if not literal_q:
        return result, enriched_args, tool_trace, 0
    distinctive = tuple(
        token
        for token in ("blush", "combo", "combopack")
        if re.search(rf"\b{token}\b", user_message, re.I)
    )
    if not distinctive or not _top_results_lack_named_tokens(result, tokens=distinctive):
        return result, enriched_args, tool_trace, 0
    current_q = str(enriched_args.get("q") or "").strip().lower()
    if current_q == literal_q.lower() and _search_has_products(result):
        return result, enriched_args, tool_trace, 0
    retry_args = _inject_tool_currency(
        SEARCH_PRODUCTS_TOOL,
        {
            **enriched_args,
            "q": literal_q,
            "category": "Flowers",
            "sort": "relevance",
        },
        state,
        currency,
    )
    if _is_duplicate_invocation(tool_trace, SEARCH_PRODUCTS_TOOL, retry_args):
        return result, enriched_args, tool_trace, 0
    _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
    retry_result = await _invoke_tool_with_rate_limit_retry(
        SEARCH_PRODUCTS_TOOL,
        retry_args,
        kapruka_service=kapruka_service,
        client_ip=rate_limit_key,
        currency=currency,
        max_retries=0,
    )
    retry_result = _curate_search_trace_result(retry_result, state=state)
    updated_trace = [
        *tool_trace,
        _tool_trace_entry(SEARCH_PRODUCTS_TOOL, retry_args, retry_result),
    ]
    if _search_has_products(retry_result):
        return retry_result, retry_args, updated_trace, 1
    return result, enriched_args, updated_trace, 1


async def _try_confident_discovery_fast_path(
    state: AgentState,
    *,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> dict[str, Any] | None:
    """Skip the Gemini planner when hybrid hints yield a reliable discovery search."""
    intent = state.get("intent")
    if intent not in ("discovery", "general"):
        return None
    if not _turn_needs_catalog(state):
        return None

    user_message = _extract_latest_user_message(state.get("messages") or [])
    if _confident_discovery_fast_path_blocked(state, user_message):
        return None
    if state.get("specificity_band") == "clarify":
        return None
    hybrid_context = state.get("hybrid_context") or {}
    if not is_confident_discovery_turn(
        user_message,
        hybrid_context,
        currency=currency,
        intent_metadata=state.get("intent_metadata"),
        state=dict(state),
    ):
        return None

    discovery_message = enrich_message_with_session_slots(user_message, dict(state))
    intent_meta = state.get("intent_metadata")
    if isinstance(intent_meta, dict) and intent_meta.get("topic_pivot"):
        discovery_message = user_message
    session_budget = state.get("session_budget_max")
    search_args = build_discovery_search_args(
        discovery_message,
        hybrid_context,
        currency=currency,
        intent_metadata=state.get("intent_metadata"),
        session_budget_max=(
            float(session_budget)
            if isinstance(session_budget, (int, float))
            and not (isinstance(intent_meta, dict) and intent_meta.get("topic_pivot"))
            else None
        ),
    )
    enriched_args = merge_planner_search_args(
        search_args,
        user_message=user_message,
        hybrid_context=hybrid_context,
        currency=currency,
        intent_metadata=state.get("intent_metadata"),
        state=dict(state),
    )
    enriched_args = _inject_tool_currency(
        SEARCH_PRODUCTS_TOOL,
        enriched_args,
        state,
        currency,
    )
    if _is_duplicate_invocation(tool_trace, SEARCH_PRODUCTS_TOOL, enriched_args):
        logger.debug("agent_loop: confident discovery fast-path duplicate search; skipping")
        return None

    logger.debug("agent_loop: confident discovery fast-path — skipping planner")
    _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
    result = await _invoke_tool_with_rate_limit_retry(
        SEARCH_PRODUCTS_TOOL,
        enriched_args,
        kapruka_service=kapruka_service,
        client_ip=rate_limit_key,
        currency=currency,
    )
    result = _curate_search_trace_result(result, state=state)
    working_trace = [
        *tool_trace,
        _tool_trace_entry(SEARCH_PRODUCTS_TOOL, enriched_args, result),
    ]
    (
        result,
        enriched_args,
        working_trace,
        color_extra,
    ) = await _maybe_retry_literal_color_flower_search(
        result,
        enriched_args,
        state=state,
        user_message=user_message,
        tool_trace=working_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    (
        result,
        enriched_args,
        working_trace,
        named_extra,
    ) = await _maybe_retry_literal_named_product_search(
        result,
        enriched_args,
        state=state,
        user_message=user_message,
        tool_trace=working_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    result, enriched_args, tool_trace, retry_calls = await _retry_anniversary_search_if_empty(
        result,
        enriched_args,
        state=state,
        user_message=user_message,
        tool_trace=working_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    retry_calls += color_extra + named_extra
    # Broaden once when primary (and anniversary) search returned 0 — strips
    # poisoned category filters like "Chocolate And Fashion".
    if not _search_has_products(result):
        intent_meta = state.get("intent_metadata")
        broadened_args, _broaden_step = apply_first_broaden(
            enriched_args,
            preserve_max_price=has_explicit_budget_constraint(
                user_message,
                state.get("session_budget_max")
                if isinstance(state.get("session_budget_max"), (int, float))
                else None,
                topic_pivot=bool(
                    isinstance(intent_meta, dict) and intent_meta.get("topic_pivot"),
                ),
            ),
            intent_metadata=intent_meta if isinstance(intent_meta, dict) else None,
        )
        if broadened_args is not None and not _is_duplicate_invocation(
            tool_trace,
            SEARCH_PRODUCTS_TOOL,
            broadened_args,
        ):
            broadened_args = _inject_tool_currency(
                SEARCH_PRODUCTS_TOOL,
                broadened_args,
                state,
                currency,
            )
            _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
            broaden_result = await _invoke_tool_with_rate_limit_retry(
                SEARCH_PRODUCTS_TOOL,
                broadened_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
            broaden_result = _curate_search_trace_result(broaden_result, state=state)
            tool_trace = [
                *tool_trace,
                {
                    "name": SEARCH_PRODUCTS_TOOL,
                    "args": broadened_args,
                    "result": broaden_result,
                },
            ]
            retry_calls += 1
            if _search_has_products(broaden_result):
                result = broaden_result
                enriched_args = broadened_args

    session_search_query_update: str | None = None
    search_q_arg = enriched_args.get("q")
    intent_meta = state.get("intent_metadata")
    if isinstance(search_q_arg, str) and search_q_arg.strip():
        session_search_query_update = _persist_session_search_query(
            search_q_arg,
            intent_metadata=intent_meta,
        )

    base_tool_count = int(state.get("tool_call_count") or 0)
    updates = _fast_path_agent_loop_updates(
        tool_trace,
        tool_call_count=base_tool_count + 1 + retry_calls,
        refined_intent="discovery",
        session_search_query_update=session_search_query_update,
    )
    last_search_products = _last_search_products_from_trace(tool_trace, state=state)
    if last_search_products:
        updates["last_search_products"] = last_search_products
    return updates


async def _try_product_detail_fast_path(
    state: AgentState,
    *,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> dict[str, Any] | None:
    """Fetch kapruka_get_product when the shopper asks about a prior carousel item."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if not is_product_detail_turn(user_message) or is_delivery_fee_question(user_message):
        return None

    for invocation in reversed(tool_trace):
        if invocation.get("name") != GET_PRODUCT_TOOL:
            continue
        if is_valid_product_detail_payload(invocation.get("result")):
            return _fast_path_agent_loop_updates(
                tool_trace,
                tool_call_count=int(state.get("tool_call_count") or 0),
            )
        break

    matched = match_product_from_last_search(
        user_message,
        state.get("last_search_products"),
        last_visible_products=state.get("last_visible_products"),
        session_product_focus=state.get("session_product_focus"),
    )
    if matched is None:
        session_resolved = state.get("session_resolved_product")
        if isinstance(session_resolved, dict) and session_resolved.get("id"):
            matched = session_resolved
    product_id = matched.get("id") if isinstance(matched, dict) else None
    if not product_id:
        return None

    args = {"product_id": str(product_id), "currency": currency}
    if _is_duplicate_invocation(tool_trace, GET_PRODUCT_TOOL, args):
        return None

    logger.debug(
        "agent_loop: product-detail fast-path — kapruka_get_product for %s",
        product_id,
    )
    _emit_status(_status_message_for_tool(GET_PRODUCT_TOOL))
    result = await _invoke_tool_with_rate_limit_retry(
        GET_PRODUCT_TOOL,
        args,
        kapruka_service=kapruka_service,
        client_ip=rate_limit_key,
        currency=currency,
    )
    tool_trace.append({"name": GET_PRODUCT_TOOL, "args": args, "result": result})
    updates = _fast_path_agent_loop_updates(
        tool_trace,
        tool_call_count=int(state.get("tool_call_count") or 0) + 1,
    )
    if is_valid_product_detail_payload(result):
        updates["session_resolved_product"] = normalize_resolved_product(result)
    return updates


async def _try_budget_refinement_fast_path(
    state: AgentState,
    *,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> dict[str, Any] | None:
    """Filter prior carousel (or MCP search) for budget-only turns — skip planner."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    focus = state.get("session_product_focus")
    focus_str = focus.strip() if isinstance(focus, str) else None
    if not is_budget_refinement_message(user_message, session_product_focus=focus_str):
        return None
    if not (
        state.get("session_search_query")
        or state.get("session_product_focus")
        or state.get("last_search_products")
        or state.get("last_visible_products")
    ):
        return None

    base_tool_count = int(state.get("tool_call_count") or 0)
    working_trace = list(tool_trace)
    tool_name = SEARCH_PRODUCTS_TOOL
    # Prefer last_visible (what the shopper saw) then last_search.
    # In-memory filter MUST run before build_budget_refinement_search_args: when args
    # cannot be built (missing q / budgeted_gift_discovery leak) we still keep
    # under-budget carousel items instead of falling through to gift-voucher search.
    prior_products = [
        item for item in list(state.get("last_visible_products") or []) if isinstance(item, dict)
    ]
    if not prior_products:
        prior_products = [
            item for item in list(state.get("last_search_products") or []) if isinstance(item, dict)
        ]
    budget_max_val = state.get("session_budget_max")
    if not isinstance(budget_max_val, (int, float)) or budget_max_val <= 0:
        from lib.neo4j.hybrid_context import extract_budget

        turn_cap = extract_budget(user_message)
        if turn_cap is not None and turn_cap.amount > 0:
            budget_max_val = float(turn_cap.amount)
        else:
            turn_budget = state.get("budget_max")
            if isinstance(turn_budget, (int, float)) and turn_budget > 0:
                budget_max_val = float(turn_budget)
    session_search_query_update: str | None = None

    budget_refinement_args = build_budget_refinement_search_args(
        dict(state),
        user_message,
        currency=currency,
    )

    if prior_products and isinstance(budget_max_val, (int, float)) and budget_max_val > 0:
        refined_in_memory = refine_last_search_by_budget(
            prior_products,
            budget_max=float(budget_max_val),
            currency=currency,
            session_product_focus=state.get("session_product_focus")
            if isinstance(state.get("session_product_focus"), str)
            else None,
            session_search_query=state.get("session_search_query")
            if isinstance(state.get("session_search_query"), str)
            else None,
            session_recipient_hint=state.get("session_recipient_hint")
            if isinstance(state.get("session_recipient_hint"), str)
            else None,
            user_message=user_message,
            hybrid_context=state.get("hybrid_context")
            if isinstance(state.get("hybrid_context"), dict)
            else None,
        )
        if not refined_in_memory:
            # Safety net when curation drops everything: keep raw under-budget picks.
            from lib.chat.product_curation import product_price_amount

            refined_in_memory = [
                product
                for product in prior_products
                if (price := product_price_amount(product)) is not None
                and price <= float(budget_max_val)
            ] or None
        if refined_in_memory and len(refined_in_memory) >= BUDGET_REFINEMENT_IN_MEMORY_MIN:
            logger.debug("agent_loop: budget refinement in-memory fast-path")
            trace_args = (
                dict(budget_refinement_args)
                if budget_refinement_args
                else {
                    "q": state.get("session_search_query") or focus_str or "gift",
                    "currency": currency,
                    "max_price": float(budget_max_val),
                    "sort": "relevance",
                }
            )
            working_trace.append(
                {
                    "name": tool_name,
                    "args": trace_args,
                    "result": {"results": refined_in_memory},
                },
            )
            _emit_provisional_carousel(
                refined_in_memory,
                state=state,
                user_message=user_message,
            )
            updates = _fast_path_agent_loop_updates(
                working_trace,
                tool_call_count=base_tool_count + 1,
            )
            updates["last_search_products"] = refined_in_memory
            updates["last_visible_products"] = refined_in_memory
            return updates

    if budget_refinement_args is None:
        return None

    enriched_args = _inject_tool_currency(
        tool_name,
        dict(budget_refinement_args),
        state,
        currency,
    )
    enriched_args = merge_planner_search_args(
        enriched_args,
        user_message=user_message,
        hybrid_context=state.get("hybrid_context") or {},
        currency=currency,
        intent_metadata=state.get("intent_metadata"),
        state=dict(state),
    )
    if prior_products:
        _emit_status(
            "None of the current picks fit that budget — searching the catalog…",
        )
    else:
        _emit_status(_status_message_for_tool(tool_name))

    result = await _invoke_tool_with_rate_limit_retry(
        tool_name,
        enriched_args,
        kapruka_service=kapruka_service,
        client_ip=rate_limit_key,
        currency=currency,
    )
    result = _curate_search_trace_result(result, state=state)
    working_trace.append(
        {"name": tool_name, "args": enriched_args, "result": result},
    )
    (
        result,
        enriched_args,
        working_trace,
        broaden_applied,
    ) = await _broaden_budget_refinement_search(
        result=result,
        enriched_args=enriched_args,
        state=state,
        tool_trace=working_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    # Primary call already appended; broaden helper appends retries only.
    search_q_arg = enriched_args.get("q")
    intent_meta = state.get("intent_metadata")
    if isinstance(search_q_arg, str) and search_q_arg.strip():
        session_search_query_update = _persist_session_search_query(
            search_q_arg,
            intent_metadata=intent_meta if isinstance(intent_meta, dict) else None,
        )
    else:
        search_q = _search_query_from_result(result)
        if search_q:
            session_search_query_update = _persist_session_search_query(
                search_q,
                intent_metadata=intent_meta if isinstance(intent_meta, dict) else None,
            )

    if _search_has_products(result):
        raw_results = result.get("results")
        products = [item for item in (raw_results or []) if isinstance(item, dict)]
        session_focus = state.get("session_product_focus")
        if (
            session_focus
            and products
            and not carousel_focus_guard(
                products,
                session_focus if isinstance(session_focus, str) else None,
            )
        ):
            demoted = demote_off_focus_products(
                products,
                session_focus if isinstance(session_focus, str) else None,
            )
            result = dict(result)
            if carousel_focus_guard(
                demoted,
                session_focus if isinstance(session_focus, str) else None,
            ):
                result["results"] = demoted
            else:
                result["results"] = demoted or products
            # Keep last trace entry in sync with demoted results.
            if working_trace:
                working_trace[-1] = {
                    **working_trace[-1],
                    "result": result,
                }

    if isinstance(result, dict) and result.get("error"):
        if _is_rate_limit_result(result) and prior_products:
            budget_label = (
                f"{int(budget_max_val):,}"
                if isinstance(budget_max_val, (int, float)) and budget_max_val > 0
                else "that"
            )
            updates = _fast_path_agent_loop_updates(
                working_trace,
                tool_call_count=base_tool_count + len(working_trace),
                exit_reason="ask_user",
            )
            updates["agent_clarifying_question"] = (
                f"None of the current picks are under Rs. {budget_label}, "
                "and our catalog is briefly busy. Please try again in a moment, "
                "or tell me a higher budget."
            )
            updates["last_search_products"] = None
            updates["last_visible_products"] = None
            return updates
        updates = _fast_path_agent_loop_updates(
            working_trace,
            tool_call_count=base_tool_count + len(working_trace),
            exit_reason="tool_error",
        )
        updates["agent_tool_error"] = _agent_tool_error_from_result(tool_name, result)
        return updates

    if _search_has_products(result):
        mcp_products = [item for item in (result.get("results") or []) if isinstance(item, dict)]
        if mcp_products:
            _emit_provisional_carousel(
                mcp_products,
                state=state,
                user_message=user_message,
            )

    logger.debug("agent_loop: budget refinement MCP fast-path")
    updates = _fast_path_agent_loop_updates(
        working_trace,
        tool_call_count=base_tool_count + len(working_trace),
        session_search_query_update=session_search_query_update,
    )
    if broaden_applied:
        updates["search_broaden_applied"] = True
    last_search_products = _last_search_products_from_trace(working_trace, state=state)
    if last_search_products:
        updates["last_search_products"] = last_search_products
    return updates


async def _try_agent_loop_fast_path(
    state: AgentState,
    *,
    tool_trace: list[ToolInvocation],
    kapruka_service: KaprukaService,
    rate_limit_key: str,
    currency: str,
) -> dict[str, Any] | None:
    """Skip the Gemini planner when intent is clear (FAQ, delivery-only, cart)."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    base_tool_count = int(state.get("tool_call_count") or 0)

    budget_fast_path = await _try_budget_refinement_fast_path(
        state,
        tool_trace=tool_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    if budget_fast_path is not None:
        return budget_fast_path

    if state.get("intent") == "cart":
        logger.debug("agent_loop: cart fast-path — skipping planner")
        return _fast_path_agent_loop_updates(
            tool_trace,
            tool_call_count=base_tool_count,
        )

    if intent_metadata.get("support_topic") or (
        is_support_question(user_message) and not _turn_needs_catalog(state)
    ):
        logger.debug("agent_loop: support FAQ fast-path — skipping planner")
        return _fast_path_agent_loop_updates(
            tool_trace,
            tool_call_count=base_tool_count,
        )

    delivery_only_session_city = state.get("session_delivery_city_canonical")
    if is_delivery_only_inquiry(
        user_message,
        intent_metadata=cast(IntentMetadata | None, intent_metadata or None),
        session_delivery_city=(
            delivery_only_session_city if isinstance(delivery_only_session_city, str) else None
        ),
    ):
        if _trace_has_check_delivery(tool_trace):
            logger.debug(
                "agent_loop: delivery-only fast-path — check_delivery already in trace",
            )
            return _fast_path_agent_loop_updates(
                tool_trace,
                tool_call_count=base_tool_count,
            )
        tool_trace, added_calls, session_delivery_date_update = await _run_pending_delivery_check(
            state,
            tool_trace,
            kapruka_service=kapruka_service,
            rate_limit_key=rate_limit_key,
            currency=currency,
            user_message=user_message,
        )
        if added_calls > 0:
            logger.debug("agent_loop: delivery-only fast-path — ran check_delivery only")
            return _fast_path_agent_loop_updates(
                tool_trace,
                tool_call_count=base_tool_count + added_calls,
                session_delivery_date_update=session_delivery_date_update,
            )

    product_detail_fast_path = await _try_product_detail_fast_path(
        state,
        tool_trace=tool_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    if product_detail_fast_path is not None:
        return product_detail_fast_path

    discovery_fast_path = await _try_confident_discovery_fast_path(
        state,
        tool_trace=tool_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    if discovery_fast_path is not None:
        return discovery_fast_path

    return None


async def agent_loop(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
    genai_client: object | None = None,
) -> dict[str, Any]:
    """LangGraph node: bounded Flash planner loop with MCP tool execution."""
    if kapruka_service is None:
        msg = "kapruka_service is required for agent_loop"
        raise ValueError(msg)

    rate_limit_key = client_ip or state.get("session_id") or "127.0.0.1"
    currency = _resolve_currency(state)

    tool_trace: list[ToolInvocation] = list(state.get("tool_trace") or [])
    tool_call_count = 0
    agent_clarifying_question: str | None = None
    agent_tool_error: dict[str, str] | None = None
    delivery_city_candidates_update: list[str] | None = None
    session_awaiting_delivery_date: bool | None = None
    agent_loop_done = False
    force_finish = False
    force_finish_reason: str | None = None
    exit_reason: str | None = None
    planner_iterations = 0
    refined_intent: Intent | None = None
    search_broaden_applied = False
    discovery_search_merged = False
    dual_gift_search_applied = False
    budgeted_gift_ideas_search_applied = False
    budget_refinement_search_applied = False
    clear_carousel_on_budget_miss = False
    situational_flowers_search_applied = False
    discovery_city_gift_search_applied = False
    session_delivery_date_update: str | None = None
    session_search_query_update: str | None = None

    user_message = _extract_latest_user_message(state.get("messages") or [])
    budget_refinement_args = build_budget_refinement_search_args(
        dict(state),
        user_message,
        currency=currency,
    )

    fast_path = await _try_agent_loop_fast_path(
        state,
        tool_trace=tool_trace,
        kapruka_service=kapruka_service,
        rate_limit_key=rate_limit_key,
        currency=currency,
    )
    if fast_path is not None:
        return fast_path

    iteration_limit = _initial_iteration_limit(state)

    for iteration in range(MAX_ITERATIONS):
        if force_finish:
            logger.debug(
                "agent_loop: %s forcing finish at iteration %s",
                force_finish_reason or "duplicate_guard",
                iteration,
            )
            exit_reason = force_finish_reason or "duplicate_guard"
            agent_loop_done = True
            break
        if iteration >= iteration_limit:
            break

        if iteration == 0 and _should_run_budgeted_gift_ideas_search(
            state,
            user_message,
            already_ran=budgeted_gift_ideas_search_applied,
        ):
            budgeted_gift_ideas_search_applied = True
            tool_name = SEARCH_PRODUCTS_TOOL
            budget_max = float(state.get("session_budget_max") or 0)
            budget_trace_start = len(tool_trace)
            base_args: dict[str, Any] = {
                "currency": currency,
                "max_price": budget_max,
                "sort": "relevance",
            }
            merged_result: dict[str, Any] = {"results": []}
            physical_queries = _budgeted_gift_physical_queries(user_message)
            _emit_status(_status_message_for_tool(tool_name))

            for physical_q in physical_queries[:BUDGETED_GIFT_SEARCH_CONCURRENCY]:
                physical_args = _inject_tool_currency(
                    tool_name,
                    {**base_args, "q": physical_q},
                    state,
                    currency,
                )
                if _is_duplicate_invocation(tool_trace, tool_name, physical_args):
                    continue
                physical_result = await _invoke_tool_with_rate_limit_retry(
                    tool_name,
                    physical_args,
                    kapruka_service=kapruka_service,
                    client_ip=rate_limit_key,
                    currency=currency,
                )
                physical_result = _curate_search_trace_result(physical_result, state=state)
                tool_trace.append(
                    {"name": tool_name, "args": physical_args, "result": physical_result},
                )
                tool_call_count += 1
                merged_result = _merge_search_results(merged_result, physical_result)
                if _search_product_count(merged_result) >= 3:
                    break

            if _search_product_count(merged_result) < 3:
                voucher_args = _inject_tool_currency(
                    tool_name,
                    {**base_args, "q": "gift voucher"},
                    state,
                    currency,
                )
                if not _is_duplicate_invocation(tool_trace, tool_name, voucher_args):
                    _emit_status(_status_message_for_tool(tool_name))
                    voucher_result = await _invoke_tool_with_rate_limit_retry(
                        tool_name,
                        voucher_args,
                        kapruka_service=kapruka_service,
                        client_ip=rate_limit_key,
                        currency=currency,
                    )
                    voucher_result = _curate_search_trace_result(voucher_result, state=state)
                    tool_trace.append(
                        {"name": tool_name, "args": voucher_args, "result": voucher_result},
                    )
                    tool_call_count += 1
                    merged_result = _merge_search_results(merged_result, voucher_result)

            session_search_query_update = physical_queries[0]
            if _search_has_products(merged_result):
                tool_trace.append(
                    {
                        "name": tool_name,
                        "args": {**base_args, "q": physical_queries[0]},
                        "result": merged_result,
                    },
                )
                tool_call_count += 1
                exit_reason = "finish"
                agent_loop_done = True
                break
            budget_searches = [
                entry for entry in tool_trace[budget_trace_start:] if entry.get("name") == tool_name
            ]
            if budget_searches and all(
                _is_rate_limit_result(entry.get("result")) for entry in budget_searches
            ):
                last_rate_limited = budget_searches[-1].get("result")
                if isinstance(last_rate_limited, dict):
                    agent_tool_error = _agent_tool_error_from_result(tool_name, last_rate_limited)
                    exit_reason = "tool_error"
                    agent_loop_done = True
                    break

        if (
            iteration == 0
            and budget_refinement_args is not None
            and not budget_refinement_search_applied
            and is_budget_refinement_message(user_message)
            and (
                state.get("session_search_query")
                or state.get("session_product_focus")
                or state.get("last_search_products")
            )
        ):
            tool_name = SEARCH_PRODUCTS_TOOL
            # Zero-MCP path: filter the prior carousel in memory when any
            # in-budget, in-focus items already exist.
            prior_products = [
                item
                for item in (
                    list(state.get("last_search_products") or [])
                    or list(state.get("last_visible_products") or [])
                )
                if isinstance(item, dict)
            ]
            budget_max_val = state.get("session_budget_max")
            refined_in_memory: list[dict[str, Any]] | None = None
            if prior_products and isinstance(budget_max_val, (int, float)) and budget_max_val > 0:
                refined_in_memory = refine_last_search_by_budget(
                    prior_products,
                    budget_max=float(budget_max_val),
                    currency=currency,
                    session_product_focus=state.get("session_product_focus")
                    if isinstance(state.get("session_product_focus"), str)
                    else None,
                    session_search_query=state.get("session_search_query")
                    if isinstance(state.get("session_search_query"), str)
                    else None,
                    session_recipient_hint=state.get("session_recipient_hint")
                    if isinstance(state.get("session_recipient_hint"), str)
                    else None,
                    user_message=user_message,
                    hybrid_context=state.get("hybrid_context")
                    if isinstance(state.get("hybrid_context"), dict)
                    else None,
                )
                if refined_in_memory and len(refined_in_memory) >= BUDGET_REFINEMENT_IN_MEMORY_MIN:
                    budget_refinement_search_applied = True
                    synthetic_args = dict(budget_refinement_args)
                    synthetic_result: dict[str, Any] = {"results": refined_in_memory}
                    tool_trace.append(
                        {
                            "name": tool_name,
                            "args": synthetic_args,
                            "result": synthetic_result,
                        },
                    )
                    tool_call_count += 1
                    _emit_provisional_carousel(
                        refined_in_memory,
                        state=state,
                        user_message=user_message,
                    )
                    exit_reason = "finish"
                    agent_loop_done = True
                    break

            enriched_args = _inject_tool_currency(
                tool_name,
                dict(budget_refinement_args),
                state,
                currency,
            )
            enriched_args = merge_planner_search_args(
                enriched_args,
                user_message=user_message,
                hybrid_context=state.get("hybrid_context") or {},
                currency=currency,
                intent_metadata=state.get("intent_metadata"),
                state=dict(state),
            )
            discovery_search_merged = True
            budget_refinement_search_applied = True
            if prior_products and refined_in_memory is None:
                _emit_status("None of the current picks fit that budget — searching the catalog…")
            result = await _invoke_tool_with_rate_limit_retry(
                tool_name,
                enriched_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
            result = _curate_search_trace_result(result, state=state)
            trace_len_before = len(tool_trace)
            tool_trace.append(
                {"name": tool_name, "args": enriched_args, "result": result},
            )
            (
                result,
                enriched_args,
                tool_trace,
                budget_broaden_applied,
            ) = await _broaden_budget_refinement_search(
                result=result,
                enriched_args=enriched_args,
                state=state,
                tool_trace=tool_trace,
                kapruka_service=kapruka_service,
                rate_limit_key=rate_limit_key,
                currency=currency,
            )
            tool_call_count += len(tool_trace) - trace_len_before
            if budget_broaden_applied:
                search_broaden_applied = True
            search_q_arg = enriched_args.get("q")
            intent_meta = state.get("intent_metadata")
            if isinstance(search_q_arg, str) and search_q_arg.strip():
                session_search_query_update = _persist_session_search_query(
                    search_q_arg,
                    intent_metadata=intent_meta,
                )
            else:
                search_q = _search_query_from_result(result)
                if search_q:
                    session_search_query_update = _persist_session_search_query(
                        search_q,
                        intent_metadata=intent_meta,
                    )
            if _search_has_products(result):
                raw_results = result.get("results")
                products = [item for item in (raw_results or []) if isinstance(item, dict)]
                session_focus = state.get("session_product_focus")
                if (
                    session_focus
                    and products
                    and not carousel_focus_guard(
                        products,
                        session_focus if isinstance(session_focus, str) else None,
                    )
                ):
                    demoted = demote_off_focus_products(
                        products,
                        session_focus if isinstance(session_focus, str) else None,
                    )
                    result = dict(result)
                    if carousel_focus_guard(
                        demoted,
                        session_focus if isinstance(session_focus, str) else None,
                    ):
                        result["results"] = demoted
                    else:
                        result["results"] = demoted or products
                    if tool_trace:
                        tool_trace[-1] = {**tool_trace[-1], "result": result}
            if isinstance(result, dict) and result.get("error"):
                # Soft miss when catalog is rate-limited after an all-over-budget carousel:
                # do not keep stale over-budget cards or surface a hard tool_error banner.
                if _is_rate_limit_result(result) and prior_products:
                    budget_label = (
                        f"{int(budget_max_val):,}"
                        if isinstance(budget_max_val, (int, float)) and budget_max_val > 0
                        else "that"
                    )
                    agent_clarifying_question = (
                        f"None of the current picks are under Rs. {budget_label}, "
                        "and our catalog is briefly busy. Please try again in a moment, "
                        "or tell me a higher budget."
                    )
                    clear_carousel_on_budget_miss = True
                    # Empty success so stale carousel fallback is skipped.
                    tool_trace[-1] = {
                        "name": tool_name,
                        "args": enriched_args,
                        "result": {"results": []},
                    }
                    exit_reason = "ask_user"
                    agent_loop_done = True
                    break
                agent_tool_error = _agent_tool_error_from_result(tool_name, result)
                exit_reason = "tool_error"
                agent_loop_done = True
                break
            if _search_has_products(result):
                mcp_products = [
                    item for item in (result.get("results") or []) if isinstance(item, dict)
                ]
                if mcp_products:
                    _emit_provisional_carousel(
                        mcp_products,
                        state=state,
                        user_message=user_message,
                    )
            if _search_has_products(result) and _should_force_finish_after_search(
                state,
                tool_trace,
            ):
                exit_reason = "finish"
                agent_loop_done = True
                break
            exit_reason = "finish"
            agent_loop_done = True
            break

        if (
            iteration == 0
            and not situational_flowers_search_applied
            and _should_run_situational_flowers_search(
                state,
                user_message,
                already_ran=situational_flowers_search_applied,
            )
        ):
            situational_flowers_search_applied = True
            tool_name = SEARCH_PRODUCTS_TOOL
            enriched_args = _inject_tool_currency(
                tool_name,
                _situational_flowers_search_args(user_message),
                state,
                currency,
            )
            _emit_status(_status_message_for_tool(tool_name))
            result = await _invoke_tool_with_rate_limit_retry(
                tool_name,
                enriched_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
            result = _curate_search_trace_result(result, state=state)
            search_q_arg = enriched_args.get("q")
            if isinstance(search_q_arg, str) and search_q_arg.strip():
                session_search_query_update = search_q_arg.strip()
            tool_trace.append(
                {"name": tool_name, "args": enriched_args, "result": result},
            )
            tool_call_count += 1
            if isinstance(result, dict) and result.get("error"):
                agent_tool_error = _agent_tool_error_from_result(tool_name, result)
                exit_reason = "tool_error"
                agent_loop_done = True
                break
            if _search_has_products(result) and _should_force_finish_after_search(
                state,
                tool_trace,
            ):
                exit_reason = "finish"
                agent_loop_done = True
                break

        if (
            iteration == 0
            and not discovery_city_gift_search_applied
            and _should_run_discovery_city_gift_search(
                state,
                user_message,
                already_ran=discovery_city_gift_search_applied,
            )
        ):
            discovery_city_gift_search_applied = True
            tool_name = SEARCH_PRODUCTS_TOOL
            enriched_args = _inject_tool_currency(
                tool_name,
                _discovery_city_gift_search_args(state, user_message),
                state,
                currency,
            )
            _emit_status(_status_message_for_tool(tool_name))
            result = await _invoke_tool_with_rate_limit_retry(
                tool_name,
                enriched_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
            result = _curate_search_trace_result(result, state=state)
            search_q_arg = enriched_args.get("q")
            if isinstance(search_q_arg, str) and search_q_arg.strip():
                session_search_query_update = _persist_session_search_query(
                    search_q_arg,
                    intent_metadata=state.get("intent_metadata"),
                )
            tool_trace.append(
                {"name": tool_name, "args": enriched_args, "result": result},
            )
            tool_call_count += 1
            if isinstance(result, dict) and result.get("error"):
                agent_tool_error = _agent_tool_error_from_result(tool_name, result)
                exit_reason = "tool_error"
                agent_loop_done = True
                break
            if _search_has_products(result) and _should_force_finish_after_search(
                state,
                tool_trace,
            ):
                exit_reason = "finish"
                agent_loop_done = True
                break

        if iteration > 0 or not budget_refinement_search_applied:
            intent_metadata = state.get("intent_metadata") or {}
            has_budget = bool(state.get("session_budget_max") or intent_metadata.get("budget_max"))
            _emit_status(
                long_search_status_message(iteration=iteration, has_budget=has_budget),
            )

        try:
            step = await _plan_next_step(
                state=state,
                tool_trace=tool_trace,
            )
        except Exception as exc:
            if not is_transient_nim_error(exc):
                raise
            logger.warning(
                "agent_loop: planner transient NIM error at iteration %s; "
                "finishing gracefully with fallback",
                iteration,
                exc_info=True,
            )
            agent_tool_error = _planner_transient_error(exc)
            exit_reason = "tool_error"
            agent_loop_done = True
            break
        planner_iterations = iteration + 1

        if iteration == 0 and step.refined_intent in ("discovery", "general"):
            refined_intent = step.refined_intent
            iteration_limit = _max_iterations_for_state(state, refined_intent)

        iteration_args: dict[str, Any] = (
            dict(step.tool_args or {}) if step.action == "call_tool" else {"action": step.action}
        )
        trace_agent_iteration(
            iteration,
            step.tool_name if step.action == "call_tool" else None,
            iteration_args,
        )
        logger.debug(
            "agent_loop: iteration %s action=%s tool=%s",
            iteration,
            step.action,
            step.tool_name,
        )

        if step.action == "ask_user":
            agent_clarifying_question = (
                step.rationale.strip() or "Could you share a few more details?"
            )
            exit_reason = "ask_user"
            agent_loop_done = True
            break

        if step.action == "finish":
            if _delivery_check_pending(state, tool_trace, user_message):
                tool_name = CHECK_DELIVERY_TOOL
                canonical_city = state.get("delivery_city_canonical") or state.get(
                    "session_delivery_city_canonical",
                )
                session_date = state.get("session_delivery_date") or state.get("delivery_date")
                delivery_args: dict[str, Any] = {
                    "city": str(canonical_city).strip(),
                    "delivery_date": str(session_date).strip(),
                }
                product_id = _resolve_delivery_product_id(state)
                if product_id:
                    delivery_args["product_id"] = product_id
                delivery_args = _inject_tool_currency(
                    tool_name,
                    delivery_args,
                    state,
                    currency,
                )
                if not _is_duplicate_invocation(tool_trace, tool_name, delivery_args):
                    _emit_status(_status_message_for_tool(tool_name))
                    delivery_result = await invoke_tool(
                        tool_name,
                        delivery_args,
                        kapruka_service=kapruka_service,
                        client_ip=rate_limit_key,
                        currency=currency,
                    )
                    tool_trace.append(
                        {
                            "name": tool_name,
                            "args": delivery_args,
                            "result": delivery_result,
                        },
                    )
                    tool_call_count += 1
                    if isinstance(delivery_result, dict) and not delivery_result.get("error"):
                        session_awaiting_delivery_date = False
                        session_delivery_date_update = delivery_args["delivery_date"]
                    continue
            exit_reason = "finish"
            agent_loop_done = True
            break

        if step.action != "call_tool":
            logger.warning("agent_loop: unknown action %r; finishing", step.action)
            exit_reason = "finish"
            agent_loop_done = True
            break

        tool_name = (step.tool_name or "").strip()
        if tool_name not in ALLOWED_PLANNER_TOOLS:
            logger.warning("agent_loop: disallowed tool %r; finishing", tool_name)
            exit_reason = "finish"
            agent_loop_done = True
            break

        raw_args = normalize_planner_tool_args(tool_name, dict(step.tool_args or {}))
        enriched_args = _inject_tool_currency(tool_name, raw_args, state, currency)

        if tool_name == GET_PRODUCT_TOOL:
            enriched_args, product_error = enrich_get_product_args(enriched_args, state)
            if product_error is not None:
                agent_tool_error = {
                    "tool": GET_PRODUCT_TOOL,
                    "message": str(product_error.get("message", "product_id_unresolved")),
                    "error": str(product_error.get("error", "product_id_unresolved")),
                }
                exit_reason = "tool_error"
                agent_loop_done = True
                break

        if tool_name == SEARCH_PRODUCTS_TOOL and not discovery_search_merged:
            enriched_args = merge_planner_search_args(
                enriched_args,
                user_message=_extract_latest_user_message(state.get("messages") or []),
                hybrid_context=state.get("hybrid_context") or {},
                currency=currency,
                intent_metadata=state.get("intent_metadata"),
                state=dict(state),
            )
            topic_pivot = bool((state.get("intent_metadata") or {}).get("topic_pivot"))
            if not topic_pivot and (
                is_broad_cakes_query(user_message)
                or is_broad_cakes_query(str(enriched_args.get("q") or ""))
            ):
                enriched_args["q"] = "birthday cake"
                enriched_args.setdefault("category", "Birthday")
            discovery_search_merged = True

        if tool_name == CHECK_DELIVERY_TOOL:
            from lib.chat.city_resolution import _is_bare_colombo, resolve_delivery_city

            user_message = _extract_latest_user_message(state.get("messages") or [])
            product_id = _resolve_delivery_product_id(state)
            if product_id and not enriched_args.get("product_id"):
                enriched_args["product_id"] = product_id
            canonical_city = state.get("delivery_city_canonical")
            if not (isinstance(canonical_city, str) and canonical_city.strip()):
                session_city = state.get("session_delivery_city_canonical")
                if isinstance(session_city, str) and session_city.strip():
                    canonical_city = session_city.strip()
            if not (isinstance(enriched_args.get("city"), str) and enriched_args["city"].strip()):
                delivery_meta = dict(state.get("intent_metadata") or {})
                target_city = delivery_meta.get("target_city")
                if isinstance(target_city, str) and target_city.strip():
                    enriched_args["city"] = target_city.strip()
            if (
                isinstance(canonical_city, str)
                and canonical_city.strip()
                and not _is_bare_colombo(canonical_city)
            ):
                enriched_args["city"] = canonical_city.strip()
            else:
                city_candidate = ""
                if isinstance(enriched_args.get("city"), str):
                    city_candidate = enriched_args["city"].strip()
                elif isinstance(canonical_city, str):
                    city_candidate = canonical_city.strip()
                if not city_candidate:
                    agent_clarifying_question = "Which city should we deliver to?"
                    exit_reason = "ask_user"
                    agent_loop_done = True
                    break
                if kapruka_service is None:
                    agent_clarifying_question = (
                        "I couldn't verify that delivery city right now. "
                        "Please try a nearby area (for example Colombo 03, Kandy, or Galle)."
                    )
                    exit_reason = "ask_user"
                    agent_loop_done = True
                    break
                resolution = await resolve_delivery_city(
                    kapruka_service,
                    rate_limit_key,
                    city_candidate,
                )
                if resolution.status == "resolved" and resolution.canonical:
                    enriched_args["city"] = resolution.canonical
                else:
                    agent_clarifying_question = (
                        resolution.customer_message or "Which city should we deliver to?"
                    )
                    if resolution.candidates:
                        delivery_city_candidates_update = list(resolution.candidates)
                    exit_reason = "ask_user"
                    agent_loop_done = True
                    break
            message_date = normalize_delivery_date({}, user_message)
            if message_date is not None:
                enriched_args["delivery_date"] = message_date
            else:
                state_date = state.get("delivery_date")
                session_date = state.get("session_delivery_date")
                if isinstance(session_date, str) and session_date.strip():
                    enriched_args["delivery_date"] = session_date.strip()
                elif isinstance(state_date, str) and state_date.strip():
                    enriched_args["delivery_date"] = state_date.strip()
            resolved_date = normalize_delivery_date(enriched_args, user_message)
            if resolved_date is None:
                if should_defer_delivery_date(state, user_message):
                    enriched_args.pop("delivery_date", None)
                    enriched_args.pop("date", None)
                else:
                    agent_clarifying_question = delivery_date_clarifying_question()
                    session_awaiting_delivery_date = True
                    exit_reason = "ask_user"
                    agent_loop_done = True
                    break
            else:
                enriched_args = {**enriched_args, "delivery_date": resolved_date}
                enriched_args.pop("date", None)

        if _is_duplicate_invocation(tool_trace, tool_name, enriched_args):
            logger.debug(
                "agent_loop: duplicate %s with identical args; forcing finish next iteration",
                tool_name,
            )
            force_finish = True
            force_finish_reason = "duplicate_guard"
            continue

        if tool_name == SEARCH_PRODUCTS_TOOL:
            result = await _invoke_tool_with_rate_limit_retry(
                tool_name,
                enriched_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
        else:
            _emit_status(_status_message_for_tool(tool_name))
            result = await invoke_tool(
                tool_name,
                enriched_args,
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                currency=currency,
            )
        if tool_name == SEARCH_PRODUCTS_TOOL:
            result = _curate_search_trace_result(result, state=state)
            search_q_arg = enriched_args.get("q")
            intent_meta = state.get("intent_metadata")
            if isinstance(search_q_arg, str) and search_q_arg.strip():
                session_search_query_update = _persist_session_search_query(
                    search_q_arg,
                    intent_metadata=intent_meta,
                )
            else:
                search_q = _search_query_from_result(result)
                if search_q:
                    session_search_query_update = _persist_session_search_query(
                        search_q,
                        intent_metadata=intent_meta,
                    )
            if _search_has_products(result):
                raw_results = result.get("results")
                products = [item for item in (raw_results or []) if isinstance(item, dict)]
                if (
                    products
                    and _accessory_ratio(products) > 0.5
                    and not bool(
                        (state.get("intent_metadata") or {}).get("topic_pivot"),
                    )
                    and (
                        is_broad_cakes_query(user_message)
                        or is_broad_cakes_query(str(enriched_args.get("q") or ""))
                    )
                ):
                    retry_args = {
                        **enriched_args,
                        "q": "birthday cake",
                        "category": "Birthday",
                    }
                    if not _is_duplicate_invocation(tool_trace, SEARCH_PRODUCTS_TOOL, retry_args):
                        _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
                        retry_result = await invoke_tool(
                            SEARCH_PRODUCTS_TOOL,
                            _inject_tool_currency(
                                SEARCH_PRODUCTS_TOOL,
                                retry_args,
                                state,
                                currency,
                            ),
                            kapruka_service=kapruka_service,
                            client_ip=rate_limit_key,
                            currency=currency,
                        )
                        retry_result = _curate_search_trace_result(
                            retry_result,
                            state=state,
                        )
                        tool_trace.append(
                            {
                                "name": SEARCH_PRODUCTS_TOOL,
                                "args": retry_args,
                                "result": retry_result,
                            },
                        )
                        tool_call_count += 1
                        if _search_has_products(retry_result):
                            result = retry_result
                            retry_q = _search_query_from_result(retry_result)
                            if retry_q:
                                session_search_query_update = _persist_session_search_query(
                                    retry_q,
                                    intent_metadata=intent_meta,
                                )
            if _should_run_dual_gift_search(
                state,
                enriched_args,
                already_ran=dual_gift_search_applied,
            ):
                dual_gift_search_applied = True
                user_message = _extract_latest_user_message(state.get("messages") or [])
                physical_args = _inject_tool_currency(
                    SEARCH_PRODUCTS_TOOL,
                    {
                        **enriched_args,
                        "q": _dual_gift_physical_query(user_message),
                    },
                    state,
                    currency,
                )
                if not _is_duplicate_invocation(tool_trace, SEARCH_PRODUCTS_TOOL, physical_args):
                    _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
                    physical_result = await invoke_tool(
                        SEARCH_PRODUCTS_TOOL,
                        physical_args,
                        kapruka_service=kapruka_service,
                        client_ip=rate_limit_key,
                        currency=currency,
                    )
                    physical_result = _curate_search_trace_result(
                        physical_result,
                        state=state,
                    )
                    tool_trace.append(
                        {
                            "name": SEARCH_PRODUCTS_TOOL,
                            "args": physical_args,
                            "result": physical_result,
                        },
                    )
                    tool_call_count += 1
                    if isinstance(physical_result, dict) and physical_result.get("error"):
                        logger.debug(
                            "agent_loop: dual gift physical search returned error %r",
                            physical_result.get("error"),
                        )
                    else:
                        result = _merge_search_results(physical_result, result)
        tool_trace.append(
            {
                "name": tool_name,
                "args": enriched_args,
                "result": result,
            },
        )
        tool_call_count += 1

        if isinstance(result, dict) and result.get("error"):
            agent_tool_error = _agent_tool_error_from_result(tool_name, result)
            exit_reason = "tool_error"
            agent_loop_done = True
            logger.debug(
                "agent_loop: tool %s returned error %r; stopping loop",
                tool_name,
                result.get("error"),
            )
            break

        if tool_name == CHECK_DELIVERY_TOOL and not (
            isinstance(result, dict) and result.get("error")
        ):
            session_awaiting_delivery_date = False
            delivery_date_arg = enriched_args.get("delivery_date")
            if isinstance(delivery_date_arg, str) and delivery_date_arg.strip():
                session_delivery_date_update = delivery_date_arg.strip()

        if (
            tool_name == SEARCH_PRODUCTS_TOOL
            and not _search_has_products(result)
            and not search_broaden_applied
        ):
            broadened_args, _broaden_step = apply_first_broaden(
                enriched_args,
                preserve_max_price=has_explicit_budget_constraint(
                    _extract_latest_user_message(state.get("messages") or []),
                    state.get("session_budget_max")
                    if isinstance(state.get("session_budget_max"), (int, float))
                    else None,
                    topic_pivot=bool(
                        (state.get("intent_metadata") or {}).get("topic_pivot"),
                    ),
                ),
                intent_metadata=state.get("intent_metadata")
                if isinstance(state.get("intent_metadata"), dict)
                else None,
            )
            if broadened_args is not None and not _is_duplicate_invocation(
                tool_trace,
                SEARCH_PRODUCTS_TOOL,
                broadened_args,
            ):
                search_broaden_applied = True
                _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
                broaden_result = await invoke_tool(
                    SEARCH_PRODUCTS_TOOL,
                    _inject_tool_currency(SEARCH_PRODUCTS_TOOL, broadened_args, state, currency),
                    kapruka_service=kapruka_service,
                    client_ip=rate_limit_key,
                    currency=currency,
                )
                broaden_result = _curate_search_trace_result(broaden_result, state=state)
                tool_trace.append(
                    {
                        "name": SEARCH_PRODUCTS_TOOL,
                        "args": broadened_args,
                        "result": broaden_result,
                    },
                )
                tool_call_count += 1
                if isinstance(broaden_result, dict) and broaden_result.get("error"):
                    agent_tool_error = _agent_tool_error_from_result(
                        SEARCH_PRODUCTS_TOOL,
                        broaden_result,
                    )
                    exit_reason = "tool_error"
                    agent_loop_done = True
                    logger.debug(
                        "agent_loop: broaden search returned error %r; stopping loop",
                        broaden_result.get("error"),
                    )
                    break
                if _search_has_products(broaden_result) and _should_force_finish_after_search(
                    state,
                    tool_trace,
                ):
                    logger.debug(
                        "agent_loop: broaden search returned products; forcing finish",
                    )
                    force_finish = True
                    force_finish_reason = "finish"
                elif not _search_has_products(broaden_result) and _has_budget_query(state):
                    logger.debug(
                        "agent_loop: budget search empty after broaden; forcing finish",
                    )
                    force_finish = True
                    force_finish_reason = "finish"
                continue

        if (
            tool_name == SEARCH_PRODUCTS_TOOL
            and _search_has_products(result)
            and _should_force_finish_after_search(state, tool_trace)
        ):
            logger.debug(
                "agent_loop: search returned products; forcing finish next iteration",
            )
            force_finish = True
            force_finish_reason = "finish"

    if not agent_loop_done and force_finish:
        exit_reason = force_finish_reason or "duplicate_guard"
        agent_loop_done = True
    elif not agent_loop_done:
        logger.debug("agent_loop: reached max iterations (%s); finishing", MAX_ITERATIONS)
        exit_reason = "max_iterations"
        agent_loop_done = True

    updates: dict[str, Any] = {
        "tool_trace": tool_trace,
        "tool_call_count": tool_call_count,
        "agent_loop_done": agent_loop_done,
        "agent_loop_exit_reason": exit_reason,
        "agent_loop_iterations": planner_iterations,
    }
    if tool_trace:
        # Mirror trace into tool_results; generate_response merges via merge_tool_trace.
        updates["tool_results"] = {
            invocation["name"]: invocation["result"] for invocation in tool_trace
        }
    if agent_clarifying_question is not None:
        updates["agent_clarifying_question"] = agent_clarifying_question
    if delivery_city_candidates_update is not None:
        updates["delivery_city_candidates"] = delivery_city_candidates_update
    if agent_tool_error is not None:
        updates["agent_tool_error"] = agent_tool_error
    if refined_intent is not None:
        updates["intent"] = refined_intent
    if search_broaden_applied:
        updates["search_broaden_applied"] = True
    if session_awaiting_delivery_date is not None:
        updates["session_awaiting_delivery_date"] = session_awaiting_delivery_date
    if session_delivery_date_update is not None:
        updates["session_delivery_date"] = session_delivery_date_update
        updates["delivery_date"] = session_delivery_date_update
    if session_search_query_update is not None:
        updates["session_search_query"] = session_search_query_update

    last_search_products = _last_search_products_from_trace(tool_trace, state=state)
    if clear_carousel_on_budget_miss:
        updates["last_search_products"] = None
        updates["last_visible_products"] = None
    elif last_search_products:
        prior = state.get("last_search_products") or []
        session_focus = state.get("session_product_focus")
        if (
            is_budget_refinement_message(user_message)
            and prior
            and isinstance(session_focus, str)
            and session_focus.strip()
            and not carousel_focus_guard(last_search_products, session_focus)
        ):
            updates["last_search_products"] = prior
        else:
            updates["last_search_products"] = last_search_products
            _emit_provisional_carousel(
                last_search_products,
                state=state,
                user_message=user_message,
            )
    elif state.get("last_search_products"):
        updates["last_search_products"] = state["last_search_products"]

    return updates
