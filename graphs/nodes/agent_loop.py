"""Bounded ReAct agent loop node — planner loop and trace summarization."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

from google import genai
from google.genai import types
from langgraph.config import get_stream_writer
from pydantic import BaseModel, ValidationError

from graphs.model_router import FLASH_MODEL
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, Intent, ToolInvocation
from lib.debug.trace import trace_agent_iteration
from lib.genai.fallback import generate_content_with_fallback
from lib.kapruka.service import KaprukaService
from lib.kapruka.tool_executor import inject_currency, invoke_tool
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.zep.memory import format_memory_facts_block

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4
PLANNER_MODEL = FLASH_MODEL

ALLOWED_PLANNER_TOOLS: frozenset[str] = frozenset(
    {
        SEARCH_PRODUCTS_TOOL,
        GET_PRODUCT_TOOL,
        LIST_CATEGORIES_TOOL,
        CHECK_DELIVERY_TOOL,
    },
)

_TOOL_STATUS_MESSAGES: dict[str, str] = {
    SEARCH_PRODUCTS_TOOL: "Searching catalog…",
    CHECK_DELIVERY_TOOL: "Checking delivery…",
    LIST_CATEGORIES_TOOL: "Browsing categories…",
    GET_PRODUCT_TOOL: "Fetching product details…",
}
_DEFAULT_STATUS_MESSAGE = "Searching catalog…"

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
- rationale: brief trace note for debugging (not shown to the customer)

Allowed tools only:
- kapruka_search_products
- kapruka_get_product
- kapruka_list_categories
- kapruka_check_delivery

Never call kapruka_track_order.

Finish rule: If products matching the user's core request have been retrieved,
action MUST be finish. Do not run auxiliary category browsing or extra searches
unless the user explicitly requested them.

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
    instruction = PLANNER_SYSTEM_INSTRUCTION
    zep_memory_facts = state.get("zep_memory_facts")
    if zep_memory_facts:
        instruction += format_memory_facts_block(zep_memory_facts)
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
    user_message = _extract_latest_user_message(state.get("messages") or [])
    return f"Customer message:\n{user_message}"


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
    for invocation in tool_trace:
        if invocation["name"] == tool_name and _args_equal(invocation["args"], tool_args):
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
    tool_call_count = state.get("tool_call_count") or 0
    agent_clarifying_question: str | None = None
    agent_loop_done = False
    force_finish = False
    exit_reason: str | None = None
    planner_iterations = 0
    refined_intent: Intent | None = None

    _emit_status(_DEFAULT_STATUS_MESSAGE)

    for iteration in range(MAX_ITERATIONS):
        if force_finish:
            logger.debug(
                "agent_loop: duplicate-tool guard forcing finish at iteration %s",
                iteration,
            )
            exit_reason = "duplicate_guard"
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

        raw_args = dict(step.tool_args or {})
        enriched_args = inject_currency(tool_name, raw_args, currency)

        if _is_duplicate_invocation(tool_trace, tool_name, enriched_args):
            logger.debug(
                "agent_loop: duplicate %s with identical args; forcing finish next iteration",
                tool_name,
            )
            force_finish = True
            continue

        _emit_status(_status_message_for_tool(tool_name))

        result = await invoke_tool(
            tool_name,
            enriched_args,
            kapruka_service=kapruka_service,
            client_ip=rate_limit_key,
            currency=currency,
        )
        tool_trace.append(
            {
                "name": tool_name,
                "args": enriched_args,
                "result": result,
            },
        )
        tool_call_count += 1

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
    if refined_intent is not None:
        updates["intent"] = refined_intent
    return updates
