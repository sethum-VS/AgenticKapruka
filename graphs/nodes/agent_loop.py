"""Bounded ReAct agent loop node — planner loop and trace summarization."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

from google import genai
from google.genai import types
from langgraph.config import get_stream_writer
from pydantic import BaseModel, ValidationError

from graphs.model_router import FLASH_MODEL
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, Intent, ToolInvocation
from lib.chat.delivery_dates import (
    delivery_date_clarifying_question,
    normalize_delivery_date,
)
from lib.chat.product_curation import (
    apply_birthday_cake_curation,
    apply_puja_curation,
    has_graph_hybrid_context,
    is_flower_fruit_intent,
)
from lib.chat.search_broadening import apply_first_broaden
from lib.debug.trace import trace_agent_iteration
from lib.genai.fallback import generate_content_with_fallback
from lib.kapruka.service import KaprukaService
from lib.kapruka.tool_executor import (
    canonical_tool_args_for_dedup,
    inject_currency,
    invoke_tool,
    normalize_planner_tool_args,
)
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.neo4j.hybrid_context import (
    is_birthday_cake_intent,
    merge_planner_search_args,
)
from lib.utils.timezone import colombo_today_iso
from lib.zep.memory import format_memory_facts_block, scope_memory_facts_for_turn

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
UTILITY_GENERAL_MAX_ITERATIONS = 2
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
    SEARCH_PRODUCTS_TOOL: "Searching Kapruka…",
    CHECK_DELIVERY_TOOL: "Checking delivery…",
    LIST_CITIES_TOOL: "Listing delivery cities…",
    LIST_CATEGORIES_TOOL: "Browsing categories…",
    GET_PRODUCT_TOOL: "Fetching product details…",
}
_DEFAULT_STATUS_MESSAGE = "Searching Kapruka…"

PLANNER_SEARCH_RESULT_LIMIT = 5
PLANNER_CATEGORY_NODE_LIMIT = 10

_SEARCH_PRODUCT_FIELDS = frozenset({"id", "name", "price", "in_stock"})
_GET_PRODUCT_FIELDS = frozenset({"id", "name", "price", "in_stock"})

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
Rs. 5,000") → action MUST be call_tool kapruka_search_products with
q="gift voucher" and max_price from their budget before ask_user.

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
_CATALOG_INTENT = re.compile(
    r"\b(?:cake|flower|chocolate|gift|hamper|bouquet|roses?|product|search|find|show|"
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
            )
            or None
        )
    return None


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
    curated = apply_birthday_cake_curation(
        apply_puja_curation(
            products,
            query=user_message,
            graph_context_available=graph_up,
        ),
        query=user_message,
        hybrid_context=hybrid_context,
        graph_context_available=graph_up,
    )
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


def _max_iterations_for_state(state: AgentState, refined_intent: Intent | None) -> int:
    """Cap planner iterations for fast utility/general turns that need no catalog."""
    if refined_intent != "general":
        return MAX_ITERATIONS
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    if intent_metadata.get("is_situational"):
        return MAX_ITERATIONS
    if _turn_needs_catalog(state):
        return MAX_ITERATIONS
    return UTILITY_GENERAL_MAX_ITERATIONS


def _should_force_finish_after_search(
    state: AgentState,
    tool_trace: list[ToolInvocation],
) -> bool:
    """Return True when a successful search should end the loop on the next iteration."""
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
) -> str:
    """Soft search-query rewrite suggestions for broad cake and mom/birthday turns."""
    hints: list[str] = []
    has_budget = budget_max is not None and budget_max > 0
    if message_count > 1 and _SHORT_CATEGORY_REPLY.match(user_message.strip()):
        hints.append(
            "Follow-up category reply after a prior clarifying turn: prefer action call_tool "
            'with kapruka_search_products (e.g. "cakes" → q="birthday cake"; '
            '"flowers" → q="fresh roses bouquet") rather than ask_user.'
        )
    if _CAKES_BROAD.search(user_message) and not _BIRTHDAY_CAKE.search(user_message):
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
                f'Budgeted "gifts" query: prefer action call_tool with kapruka_search_products '
                f'q="gift voucher" and max_price={budget_max} rather than ask_user.'
            )
        else:
            hints.append(
                'Vague "gifts" query with no occasion, recipient, or budget: prefer action '
                "ask_user before kapruka_search_products."
            )
    elif has_budget and _GIFT_WORD.search(user_message):
        hints.append(
            f"Budgeted gift query: prefer action call_tool with kapruka_search_products "
            f'q="gift voucher" and max_price={budget_max} before ask_user.'
        )
    if _FLOWERS_REQUEST.search(user_message):
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
    instruction = (
        f"{PLANNER_SYSTEM_INSTRUCTION}\n\n"
        f"Today in Sri Lanka: {colombo_today_iso()}\n"
        "For kapruka_check_delivery, delivery_date must be YYYY-MM-DD on or after today."
    )
    zep_memory_facts = state.get("zep_memory_facts")
    if zep_memory_facts:
        user_message = _extract_latest_user_message(state.get("messages") or [])
        scoped_facts = scope_memory_facts_for_turn(zep_memory_facts, user_message)
        if scoped_facts:
            instruction += format_memory_facts_block(scoped_facts)
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
    intent_metadata: dict[str, Any] = dict(state.get("intent_metadata") or {})
    budget_max = intent_metadata.get("budget_max")
    hybrid_context = state.get("hybrid_context") or {}
    rewrite_hints = _format_planner_query_rewrite_hints(
        user_message,
        message_count=len(messages),
        budget_max=budget_max if isinstance(budget_max, (int, float)) else None,
        graph_context_available=has_graph_hybrid_context(hybrid_context),
    )
    if rewrite_hints:
        prompt += f"\n\n{rewrite_hints}"
    return prompt


def _parse_planner_step(response: types.GenerateContentResponse) -> AgentPlannerStep:
    """Parse structured or JSON text planner step from a Gemini response."""
    if response.parsed is not None:
        if isinstance(response.parsed, AgentPlannerStep):
            return response.parsed
        return AgentPlannerStep.model_validate(response.parsed)

    raw_text = (response.text or "").strip()
    if not raw_text:
        msg = "Gemini returned empty planner step"
        raise ValueError(msg)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"Gemini planner response is not valid JSON: {raw_text!r}"
        raise ValueError(msg) from exc

    try:
        return AgentPlannerStep.model_validate(payload)
    except ValidationError as exc:
        msg = f"Gemini planner JSON failed validation: {payload!r}"
        raise ValueError(msg) from exc


def _plan_next_step_sync(
    client: genai.Client | None,
    *,
    state: AgentState,
    tool_trace: list[ToolInvocation],
) -> AgentPlannerStep:
    """Blocking Gemini planner call; run via asyncio.to_thread from agent_loop."""
    response = generate_content_with_fallback(
        client=client,
        model=PLANNER_MODEL,
        contents=_build_planner_user_prompt(state),
        config=types.GenerateContentConfig(
            system_instruction=_build_planner_system_instruction(state, tool_trace=tool_trace),
            response_mime_type="application/json",
            response_schema=AgentPlannerStep,
            temperature=0,
        ),
    )
    return _parse_planner_step(response)


def _emit_status(message: str) -> None:
    """Emit a LangGraph custom stream status event when a writer is available."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    if writer is not None:
        writer({"type": "status", "message": message})


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


async def agent_loop(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
    genai_client: genai.Client | None = None,
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
    session_awaiting_delivery_date: bool | None = None
    agent_loop_done = False
    force_finish = False
    force_finish_reason: str | None = None
    exit_reason: str | None = None
    planner_iterations = 0
    refined_intent: Intent | None = None
    search_broaden_applied = False
    discovery_search_merged = False

    _emit_status(_DEFAULT_STATUS_MESSAGE)

    iteration_limit = MAX_ITERATIONS

    for iteration in range(MAX_ITERATIONS):
        if iteration >= iteration_limit:
            break
        if force_finish:
            logger.debug(
                "agent_loop: %s forcing finish at iteration %s",
                force_finish_reason or "duplicate_guard",
                iteration,
            )
            exit_reason = force_finish_reason or "duplicate_guard"
            agent_loop_done = True
            break

        _emit_status(_DEFAULT_STATUS_MESSAGE)

        step = await asyncio.to_thread(
            _plan_next_step_sync,
            genai_client,
            state=state,
            tool_trace=tool_trace,
        )
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
        enriched_args = inject_currency(tool_name, raw_args, currency)

        if tool_name == SEARCH_PRODUCTS_TOOL and not discovery_search_merged:
            enriched_args = merge_planner_search_args(
                enriched_args,
                user_message=_extract_latest_user_message(state.get("messages") or []),
                hybrid_context=state.get("hybrid_context") or {},
                currency=currency,
                intent_metadata=state.get("intent_metadata"),
            )
            discovery_search_merged = True

        if tool_name == CHECK_DELIVERY_TOOL:
            user_message = _extract_latest_user_message(state.get("messages") or [])
            canonical_city = state.get("delivery_city_canonical")
            if not (isinstance(canonical_city, str) and canonical_city.strip()):
                session_city = state.get("session_delivery_city_canonical")
                if isinstance(session_city, str) and session_city.strip():
                    canonical_city = session_city.strip()
            if isinstance(canonical_city, str) and canonical_city.strip():
                enriched_args["city"] = canonical_city.strip()
            state_date = state.get("delivery_date")
            if isinstance(state_date, str) and state_date.strip():
                enriched_args["delivery_date"] = state_date.strip()
            resolved_date = normalize_delivery_date(enriched_args, user_message)
            if resolved_date is None:
                agent_clarifying_question = delivery_date_clarifying_question()
                session_awaiting_delivery_date = True
                exit_reason = "ask_user"
                agent_loop_done = True
                break
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
        tool_trace.append(
            {
                "name": tool_name,
                "args": enriched_args,
                "result": result,
            },
        )
        tool_call_count += 1

        if isinstance(result, dict) and result.get("error"):
            error_message = result.get("message")
            agent_tool_error = {
                "tool": tool_name,
                "message": (
                    str(error_message).strip()
                    if isinstance(error_message, str) and error_message.strip()
                    else str(result.get("error"))
                ),
            }
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

        if (
            tool_name == SEARCH_PRODUCTS_TOOL
            and not _search_has_products(result)
            and not search_broaden_applied
        ):
            broadened_args, _broaden_step = apply_first_broaden(enriched_args)
            if broadened_args is not None and not _is_duplicate_invocation(
                tool_trace,
                SEARCH_PRODUCTS_TOOL,
                broadened_args,
            ):
                search_broaden_applied = True
                _emit_status(_status_message_for_tool(SEARCH_PRODUCTS_TOOL))
                broaden_result = await invoke_tool(
                    SEARCH_PRODUCTS_TOOL,
                    broadened_args,
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
                    error_message = broaden_result.get("message")
                    agent_tool_error = {
                        "tool": SEARCH_PRODUCTS_TOOL,
                        "message": (
                            str(error_message).strip()
                            if isinstance(error_message, str) and error_message.strip()
                            else str(broaden_result.get("error"))
                        ),
                    }
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

    if not agent_loop_done:
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
    if agent_tool_error is not None:
        updates["agent_tool_error"] = agent_tool_error
    if refined_intent is not None:
        updates["intent"] = refined_intent
    if search_broaden_applied:
        updates["search_broaden_applied"] = True
    if session_awaiting_delivery_date is not None:
        updates["session_awaiting_delivery_date"] = session_awaiting_delivery_date

    last_search_products = _last_search_products_from_trace(tool_trace, state=state)
    if last_search_products:
        updates["last_search_products"] = last_search_products

    return updates
