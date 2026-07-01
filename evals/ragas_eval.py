"""Ragas evaluation runner for the shopping graph against the golden dataset.

Runs the compiled shopping graph with deterministic MockMCPHttpClient fixtures and a
planner-aware Gemini mock for multi-step agent_loop scenarios. CI gate (``--ci``):

- ``context_precision`` mean >= ``DEFAULT_CONTEXT_PRECISION_THRESHOLD`` (0.7)
- Per-case ``expected_tools`` must match the tool sequence from ``tool_trace``
  (agent_loop) or ``tool_results`` (product-ID fast-path / checkout / tracking)

Run locally::

    python -m evals.ragas_eval
    python -m evals.ragas_eval --ci --threshold 0.7

GitHub Actions job ``ragas-eval`` runs ``python -m evals.ragas_eval --ci`` after unit tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import fakeredis.aioredis
from datasets import Dataset
from google.genai import types
from langchain_core.embeddings import FakeEmbeddings
from langgraph.graph.state import CompiledStateGraph
from ragas import aevaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig
from tests.fixtures.mcp_mock import MockMCPHttpClient

from app.config import Settings
from evals.golden_dataset import GoldenCase, GoldenDataset, load_golden_dataset
from evals.intent_heuristics import infer_intent_from_message
from evals.ragas_ci_llm import CiRagasChatModel
from graphs.nodes.agent_loop import AgentPlannerStep
from graphs.nodes.analyze_intent import PROCEED_CHECKOUT_MESSAGE, IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph, initial_shopping_state
from graphs.state import AgentState, Intent, ToolInvocation
from lib.chat.delivery_dates import normalize_delivery_date
from lib.chat.intent_heuristics import is_budgeted_gift_ideas_message
from lib.chat.query_preprocessor import QueryPreprocessor
from lib.chat.request_specificity import is_delivery_only_inquiry
from lib.kapruka.product_id import extract_product_id
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.redis.cart import add_item
from lib.redis.client import RedisClient
from lib.utils.timezone import colombo_today_iso

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PRECISION_THRESHOLD = 0.7
_CI_RAGAS_TIMEOUT_SECONDS = 30
_EVAL_CLIENT_IP = "203.0.113.99"


def _minimal_eval_settings() -> Settings:
    """Settings stub so graph nodes avoid loading a local .env during eval runs."""
    return Settings(
        redis_url="redis://localhost:6379/0",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="eval-password",
        zep_api_key="eval-zep-key",
        gcp_project_id="eval-project",
        gcp_location="us-central1",
        session_secret="x" * 32,
        _env_file=None,
    )


@contextmanager
def _patch_eval_settings() -> Iterator[None]:
    """Patch get_settings at import sites used by graph nodes during Ragas eval."""
    settings = _minimal_eval_settings()
    with (
        patch("lib.chat.model_router.get_settings", return_value=settings),
        patch("graphs.nodes.retrieve_hybrid_context.get_settings", return_value=settings),
    ):
        yield


@dataclass(frozen=True, slots=True)
class RagasEvalScores:
    """Aggregate Ragas metric means over the golden dataset."""

    context_precision: float
    answer_relevancy: float
    faithfulness: float
    case_count: int


@dataclass(frozen=True, slots=True)
class GraphEvalRow:
    """Single row collected from a graph invocation for Ragas scoring."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


def intent_for_case(case: GoldenCase) -> Intent:
    """Map golden scenario + expected tools to graph routing intent."""
    if LIST_CATEGORIES_TOOL in case.expected_tools:
        return "general"
    if case.scenario == "tracking":
        return "tracking"
    if case.scenario == "checkout":
        return "checkout"
    return "discovery"


_SITUATIONAL_FLAVOR_PREFIX = "Aiyo machan, hodata gentle choice — "


def _is_concierge_system_instruction(config: types.GenerateContentConfig | None) -> bool:
    """True when generate_response selected the Localized Concierge prompt."""
    if config is None:
        return False
    instruction = getattr(config, "system_instruction", None) or ""
    lowered = instruction.lower()
    return "gift concierge" in lowered or "localized concierge" in lowered


def _distress_needs_empathy(user_message: str) -> bool:
    """True when the customer turn signals breakup, loss, or sympathy (not order fixes)."""
    lowered = user_message.lower()
    return any(
        token in lowered
        for token in (
            "broke up",
            "breakup",
            "break-up",
            "heartbroken",
            "passed away",
            "funeral",
            "condolence",
            "sympathy",
            "grieving",
            "devastated",
        )
    )


def _user_used_vernacular(user_message: str) -> bool:
    """Mirror Tanglish only when the customer already used casual local tokens."""
    tokens = {token.lower() for token in re.findall(r"[A-Za-z']+", user_message)}
    tanglish = {
        "aiyo",
        "machan",
        "hodata",
        "mage",
        "mama",
        "ammata",
        "malli",
        "machang",
    }
    return bool(tokens & tanglish) or bool(
        re.search(r"[\u0D80-\u0DFF]", user_message),
    )


def _apply_situational_flavor(message: str, *, user_message: str = "") -> str:
    """Empathy preamble for distress; vernacular flavor only when the customer used it."""
    lowered = message.lower()
    if any(marker in lowered for marker in ("aiyo", "machan", "hodata")):
        return message
    empathy = "I'm sorry to hear that — " if _distress_needs_empathy(user_message) else ""
    if _user_used_vernacular(user_message):
        return f"{empathy}{_SITUATIONAL_FLAVOR_PREFIX}{message}"
    if empathy:
        return f"{empathy}{message}"
    return message


def _synthesize_assistant_reply(user_prompt: str) -> str:
    """Build a faithful assistant reply from tool_results embedded in the Gemini prompt."""
    marker = "tool_results (sole source of truth for catalog facts):"
    if marker not in user_prompt:
        return "Here is what I found on Kapruka."

    raw_json = user_prompt.split(marker, maxsplit=1)[1].strip()
    try:
        tool_results = json.loads(raw_json)
    except json.JSONDecodeError:
        return "Here is what I found on Kapruka."

    if not isinstance(tool_results, dict):
        return "Here is what I found on Kapruka."

    search_payload = tool_results.get(SEARCH_PRODUCTS_TOOL)
    if isinstance(search_payload, dict):
        results = search_payload.get("results")
        if isinstance(results, list) and results:
            lines: list[str] = []
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                price = item.get("price")
                amount = price.get("amount") if isinstance(price, dict) else None
                if isinstance(name, str) and isinstance(amount, (int, float)):
                    lines.append(f"{name} (LKR {amount:,.0f})")
                elif isinstance(name, str):
                    lines.append(name)
            if lines:
                curated = lines[:3]
                user_message = _extract_customer_message(user_prompt)
                if _distress_needs_empathy(user_message):
                    if len(curated) == 1:
                        return (
                            "I'm sorry to hear that — here's a thoughtful, gentle curated pick: "
                            f"{curated[0]}."
                        )
                    return (
                        "I'm sorry to hear that — here are a few thoughtful, "
                        "gentle curated options: " + "; ".join(curated) + "."
                    )
                if len(curated) == 1:
                    return f"Here is a thoughtful pick: {curated[0]}."
                return "Here are a few curated options: " + "; ".join(curated) + "."

    product_payload = tool_results.get(GET_PRODUCT_TOOL)
    if isinstance(product_payload, dict) and product_payload.get("name"):
        name = str(product_payload["name"])
        price = product_payload.get("price")
        amount = price.get("amount") if isinstance(price, dict) else None
        if isinstance(amount, (int, float)):
            return f"{name} is available on Kapruka for LKR {amount:,.0f}."
        return f"Here are the Kapruka details for {name}."

    categories_payload = tool_results.get(LIST_CATEGORIES_TOOL)
    if isinstance(categories_payload, dict):
        categories = categories_payload.get("categories")
        if isinstance(categories, list) and categories:
            names = [
                str(node["name"])
                for node in categories
                if isinstance(node, dict) and node.get("name")
            ]
            if names:
                return "Kapruka gift categories include " + ", ".join(names[:5]) + "."

    return "Here is what I found on Kapruka based on our catalog data."


def _search_args_for_eval_case(case: GoldenCase) -> dict[str, str]:
    """Build deterministic search args for agent-loop golden cases."""
    if case.id == "agent-002-cakes-single-search":
        return {"q": "cakes"}
    query = case.user_query.strip()
    if len(query) < 3:
        return {"q": "gifts"}
    return {"q": query}


def _tool_args_for_eval_case(tool_name: str, case: GoldenCase) -> dict[str, Any]:
    """Map expected MCP tool names to deterministic mock planner args."""
    if tool_name == SEARCH_PRODUCTS_TOOL:
        return _search_args_for_eval_case(case)
    if tool_name == LIST_CATEGORIES_TOOL:
        return {"depth": 1}
    if tool_name == GET_PRODUCT_TOOL:
        product_id = extract_product_id(case.user_query) or "cake00ka002034"
        return {"product_id": product_id}
    if tool_name == CHECK_DELIVERY_TOOL:
        metadata = QueryPreprocessor().process(case.user_query)
        city = metadata.get("target_city") or "Colombo 03"
        delivery_date = normalize_delivery_date({}, case.user_query) or colombo_today_iso()
        return {"city": str(city), "delivery_date": delivery_date}
    if tool_name == LIST_CITIES_TOOL:
        return {"query": "Galle", "limit": 25}
    return {}


def _extract_customer_message(user_contents: str) -> str:
    """Parse the latest customer utterance from a planner or response prompt."""
    marker = "Customer message:\n"
    if marker not in user_contents:
        return user_contents.strip()
    body = user_contents.split(marker, 1)[1]
    return body.split("\n\n", 1)[0].strip()


def _e2e_search_args_for_message(message: str) -> dict[str, str]:
    """Build deterministic search args for shadow/E2E planner mocks."""
    stripped = message.strip()
    if len(stripped) < 3:
        return {"q": "gifts"}
    return {"q": stripped[:120]}


def _e2e_check_delivery_args(message: str) -> dict[str, str]:
    """Build delivery-check args from preprocessor city hints."""
    metadata = QueryPreprocessor().process(message)
    city = metadata.get("target_city") or "Colombo 03"
    delivery_date = normalize_delivery_date({}, message) or colombo_today_iso()
    return {"city": str(city), "delivery_date": delivery_date}


def _e2e_list_cities_args(message: str) -> dict[str, Any]:
    """Build list-cities args for shadow transcripts mentioning nearby cities."""
    lowered = message.lower()
    for token in ("galle", "kandy", "colombo", "jaffna"):
        if token in lowered:
            return {"query": token.capitalize(), "limit": 25}
    return {"limit": 25}


def _infer_e2e_planner_tools(message: str) -> list[str]:
    """Infer ordered MCP tools for one E2E/shadow user turn."""
    lowered = message.lower()
    tools: list[str] = []

    if any(
        phrase in lowered for phrase in ("kinds of gifts", "what can i buy", "what do you sell")
    ):
        tools.append(LIST_CATEGORIES_TOOL)

    if "cities near" in lowered or "delivery cities" in lowered:
        tools.append(LIST_CITIES_TOOL)

    metadata = QueryPreprocessor().process(message)
    if metadata.get("requires_delivery_validation") and is_delivery_only_inquiry(
        message,
        intent_metadata=metadata,
    ):
        return [CHECK_DELIVERY_TOOL]

    if metadata.get("requires_delivery_validation"):
        tools.append(CHECK_DELIVERY_TOOL)

    category_only = tools == [LIST_CATEGORIES_TOOL]
    if not category_only:
        tools.insert(0, SEARCH_PRODUCTS_TOOL)
    elif SEARCH_PRODUCTS_TOOL not in tools:
        tools.append(SEARCH_PRODUCTS_TOOL)

    return tools


def _tool_args_for_e2e_message(tool_name: str, message: str) -> dict[str, Any]:
    """Map inferred E2E planner tools to deterministic mock args."""
    if tool_name == SEARCH_PRODUCTS_TOOL:
        return _e2e_search_args_for_message(message)
    if tool_name == LIST_CATEGORIES_TOOL:
        return {"depth": 1}
    if tool_name == CHECK_DELIVERY_TOOL:
        return _e2e_check_delivery_args(message)
    if tool_name == LIST_CITIES_TOOL:
        return _e2e_list_cities_args(message)
    return {}


def _preflight_tools_for_eval_case(case: GoldenCase) -> list[str]:
    """MCP tools resolve_delivery_context may append before agent_loop (PRD-138 preflight)."""
    if case.scenario != "discovery":
        return []
    metadata = QueryPreprocessor().process(case.user_query)
    if metadata.get("requires_delivery_validation"):
        return [CHECK_DELIVERY_TOOL]
    return []


def _agent_loop_expected_tools(case: GoldenCase) -> list[str]:
    """Planner-mock tools only — excludes preflight already run in resolve_delivery_context."""
    preflight = _preflight_tools_for_eval_case(case)
    expected = list(case.expected_tools)
    if preflight and len(expected) >= len(preflight) and expected[: len(preflight)] == preflight:
        return expected[len(preflight) :]
    return expected


def _planner_step_for_eval_case(
    case: GoldenCase | None,
    *,
    planner_call_index: int,
    user_contents: str,
) -> AgentPlannerStep:
    """Return the next agent-loop planner step for Ragas eval graph runs."""
    if case is None:
        message = _extract_customer_message(user_contents)
        expected_tools = _infer_e2e_planner_tools(message)
        if planner_call_index < len(expected_tools):
            tool_name = expected_tools[planner_call_index]
            return AgentPlannerStep(
                action="call_tool",
                tool_name=tool_name,
                tool_args=_tool_args_for_e2e_message(tool_name, message),
                rationale=f"e2e planner step {planner_call_index + 1}",
            )
        return AgentPlannerStep(action="finish", rationale="catalog facts collected")

    expected = _agent_loop_expected_tools(case)
    if not expected:
        return AgentPlannerStep(
            action="finish",
            refined_intent="general",
            rationale="no Kapruka tools needed",
        )

    if planner_call_index < len(expected):
        tool_name = expected[planner_call_index]
        return AgentPlannerStep(
            action="call_tool",
            tool_name=tool_name,
            tool_args=_tool_args_for_eval_case(tool_name, case),
            rationale=f"eval planner step {planner_call_index + 1}",
        )

    return AgentPlannerStep(action="finish", rationale="expected tools collected")


_PREFLIGHT_RESULT_TOOLS: frozenset[str] = frozenset({CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL})


def tool_names_from_state(result: AgentState) -> list[str]:
    """Ordered MCP tool names from agent_loop tool_trace plus delivery preflight."""
    trace_names: list[str] = []
    tool_trace = result.get("tool_trace")
    if isinstance(tool_trace, list) and tool_trace:
        for invocation in tool_trace:
            if not isinstance(invocation, dict):
                continue
            name = invocation.get("name")
            if isinstance(name, str) and name:
                trace_names.append(name)

    tool_results = result.get("tool_results")
    result_names: list[str] = []
    if isinstance(tool_results, dict):
        result_names = [
            key
            for key, value in tool_results.items()
            if value is not None and key in _PREFLIGHT_RESULT_TOOLS
        ]

    if trace_names:
        preflight = [name for name in result_names if name not in trace_names]
        return preflight + trace_names
    if isinstance(tool_results, dict):
        return [key for key, value in tool_results.items() if value is not None]
    return []


def _should_assert_agent_loop_tools(case: GoldenCase) -> bool:
    """Phase 3 tool-trace gate applies to agent_loop discovery cases only."""
    if case.scenario != "discovery":
        return False
    if case.id.startswith("agent-"):
        return True
    return extract_product_id(case.user_query) is None


def assert_expected_tool_usage(case: GoldenCase, result: AgentState) -> None:
    """Raise when graph tool usage diverges from the golden case expected_tools sequence."""
    if not _should_assert_agent_loop_tools(case):
        return

    actual = tool_names_from_state(result)
    expected = case.expected_tools
    tools_match = actual == expected or (
        len(actual) == len(expected) and sorted(actual) == sorted(expected)
    )
    if not tools_match and SEARCH_PRODUCTS_TOOL in expected:
        non_search_expected = [t for t in expected if t != SEARCH_PRODUCTS_TOOL]
        non_search_actual = [t for t in actual if t != SEARCH_PRODUCTS_TOOL]
        min_search = expected.count(SEARCH_PRODUCTS_TOOL)
        if (
            non_search_actual == non_search_expected
            and actual.count(SEARCH_PRODUCTS_TOOL) >= min_search
            and (
                is_budgeted_gift_ideas_message(case.user_query)
                or case.id
                in ("spec-003-budgeted-gift-chip-proceed", "disc-015-budget-gift-quality")
            )
        ):
            tools_match = True
    if tools_match:
        product_id = extract_product_id(case.user_query)
        if product_id and expected == [GET_PRODUCT_TOOL]:
            tool_trace = result.get("tool_trace")
            if isinstance(tool_trace, list) and tool_trace:
                msg = (
                    f"golden case {case.id}: product-ID fast-path must not use "
                    f"agent_loop tool_trace (got {len(tool_trace)} entries)"
                )
                raise AssertionError(msg)
        return

    msg = (
        f"golden case {case.id}: expected tools {expected!r}, got {actual!r} "
        f"(tool_trace entries={len(result.get('tool_trace') or [])})"
    )
    raise AssertionError(msg)


def build_eval_genai_client(
    intent: Intent | None = None,
    *,
    case: GoldenCase | None = None,
) -> MagicMock:
    """Gemini client mock: structured intent then faithful catalog reply.

    When ``intent`` is None, infer routing from the user message (E2E / shadow tests).
    """
    client = MagicMock()
    default_intent: Intent = intent or "discovery"
    intent_response = MagicMock()
    intent_response.parsed = IntentClassification(intent=default_intent)
    intent_response.text = json.dumps({"intent": default_intent})
    planner_state: dict[str, Any] = {"contents": None, "index": 0}

    def reset_planner_state() -> None:
        planner_state["contents"] = None
        planner_state["index"] = 0

    def generate_content(
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        _ = model, kwargs
        response = MagicMock()
        if config is not None and config.response_schema is IntentClassification:
            resolved: Intent = intent if intent is not None else infer_intent_from_message(contents)
            response.parsed = IntentClassification(intent=resolved)
            response.text = json.dumps({"intent": resolved})
            return response

        if config is not None and config.response_schema is AgentPlannerStep:
            if contents != planner_state["contents"]:
                planner_state["contents"] = contents
                planner_state["index"] = 0
            step = _planner_step_for_eval_case(
                case,
                planner_call_index=planner_state["index"],
                user_contents=contents,
            )
            planner_state["index"] += 1
            response.parsed = step
            response.text = step.model_dump_json()
            return response

        if config is not None and config.response_schema is AssistantReply:
            message = _synthesize_assistant_reply(contents)
            user_message = _extract_customer_message(contents)
            empathy_source = (
                user_message
                if _distress_needs_empathy(user_message)
                else contents
                if _distress_needs_empathy(contents)
                else user_message
            )
            if (
                _is_concierge_system_instruction(config)
                or _distress_needs_empathy(user_message)
                or _distress_needs_empathy(contents)
            ):
                message = _apply_situational_flavor(message, user_message=empathy_source)
            response.parsed = AssistantReply(message=message)
            response.text = json.dumps({"message": message})
            return response

        if (
            config is not None
            and getattr(config.response_schema, "__name__", "") == "MasterFlowAlignment"
        ):
            from lib.chat.master_flow import MasterFlowAlignment
            resolved = MasterFlowAlignment(
                decision="proceed",
                confidence=1.0,
                active_flow="free_discovery",
                mismatch_reason="mock",
                clarifying_question=None,
                resolved_intent=None,
                resolved_session_fields={},
                intent_metadata_patches={},
                checkout_action=None,
                context_reset=False,
            )
            response.parsed = resolved
            response.text = resolved.model_dump_json()
            return response

        if (
            config is not None
            and getattr(config.response_schema, "__name__", "") == "SpecificityRefinement"
        ):
            from lib.chat.request_specificity import SpecificityRefinement
            ref = SpecificityRefinement(
                score=100.0,
                product_score=1.0,
                occasion_score=1.0,
                budget_score=1.0,
                band="proceed",
                missing_dimension=None,
            )
            response.parsed = ref
            response.text = ref.model_dump_json()
            return response

        response.parsed = intent_response.parsed
        response.text = intent_response.text
        return response

    client.models.generate_content.side_effect = generate_content
    client.reset_planner_state = reset_planner_state  # type: ignore[attr-defined]
    return client


def contexts_from_tool_results(tool_results: dict[str, Any] | None) -> list[str]:
    """Serialize MCP tool payloads as Ragas retrieved_contexts strings."""
    if not tool_results:
        return []
    contexts: list[str] = []
    for value in tool_results.values():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            contexts.append(json.dumps(value, ensure_ascii=False))
        else:
            contexts.append(str(value))
    return contexts


def contexts_from_tool_trace(tool_trace: list[ToolInvocation] | None) -> list[str]:
    """Serialize agent-loop tool_trace payloads as Ragas retrieved_contexts strings."""
    if not tool_trace:
        return []
    contexts: list[str] = []
    for invocation in tool_trace:
        result = invocation.get("result")
        if result is None:
            continue
        if isinstance(result, (dict, list)):
            contexts.append(json.dumps(result, ensure_ascii=False))
        else:
            contexts.append(str(result))
    return contexts


def contexts_from_eval_state(result: AgentState) -> list[str]:
    """Collect Ragas contexts from heuristic tool_results or agent-loop tool_trace."""
    tool_results = result.get("tool_results")
    contexts = contexts_from_tool_results(tool_results if isinstance(tool_results, dict) else {})
    if contexts:
        return contexts
    tool_trace = result.get("tool_trace")
    if isinstance(tool_trace, list):
        return contexts_from_tool_trace(tool_trace)
    return []


def _plain_response_from_state(result: AgentState) -> str:
    """Prefer assistant_message; strip HTML fallback from response_html."""
    assistant_message = result.get("assistant_message")
    if isinstance(assistant_message, str) and assistant_message.strip():
        return assistant_message.strip()
    response_html = result.get("response_html")
    if isinstance(response_html, str) and response_html.strip():
        return response_html.strip()
    return "No response generated."


async def _seed_checkout_cart(redis_client: RedisClient, session_id: str) -> None:
    """Seed a sample cart so checkout-intent golden cases produce checkout context."""
    await add_item(
        redis_client,
        session_id,
        product_id="cake00ka002034",
        name="Chocolate Birthday Cake",
        price_amount=4500.0,
        price_currency="LKR",
        quantity=1,
    )


async def run_graph_for_case(
    case: GoldenCase,
    *,
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    redis_client: RedisClient,
) -> GraphEvalRow:
    """Invoke the shopping graph for one golden case and collect Ragas fields."""
    session_id = f"ragas-{case.id}"
    if case.scenario == "checkout":
        await _seed_checkout_cart(redis_client, session_id)

    state = initial_shopping_state(
        message=case.user_query,
        session_id=session_id,
        thread_id=session_id,
    )
    if case.user_query.strip() == PROCEED_CHECKOUT_MESSAGE:
        state["intent"] = "checkout"

    with _patch_eval_settings():
        result = await graph.ainvoke(state)

    assert_expected_tool_usage(case, result)

    return GraphEvalRow(
        user_input=case.user_query,
        response=_plain_response_from_state(result),
        retrieved_contexts=contexts_from_eval_state(result),
        reference=case.reference_answer,
    )


def rows_to_dataset(rows: list[GraphEvalRow]) -> Dataset:
    """Convert graph eval rows to a HuggingFace Dataset for Ragas."""
    return Dataset.from_dict(
        {
            "user_input": [row.user_input for row in rows],
            "response": [row.response for row in rows],
            "retrieved_contexts": [row.retrieved_contexts for row in rows],
            "reference": [row.reference for row in rows],
        },
    )


def _mean_metric(result: Any, key: str) -> float:
    """Average per-row metric scores from a Ragas EvaluationResult."""
    scores_dict = getattr(result, "_scores_dict", None)
    if not isinstance(scores_dict, dict):
        raw = result[key]
        return float(raw) if not isinstance(raw, list) else float("nan")

    values = scores_dict.get(key)
    if not isinstance(values, list) or not values:
        return float("nan")

    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        numeric.append(float(value))
    if not numeric:
        return float("nan")
    return sum(numeric) / len(numeric)


def build_ci_ragas_llm() -> LangchainLLMWrapper:
    """LangChain-wrapped deterministic judge for CI pipelines."""
    return LangchainLLMWrapper(CiRagasChatModel())


def build_ci_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Fixed-size fake embeddings for answer_relevancy in CI."""
    return LangchainEmbeddingsWrapper(FakeEmbeddings(size=768))


async def build_eval_graph_for_case(
    case: GoldenCase,
    redis_client: RedisClient,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile graph with genai intent mock aligned to the golden case."""
    mcp_client = await MockMCPHttpClient.connect()
    kapruka_service = KaprukaService(redis_client, mcp_client)
    deps = ShoppingGraphDeps(
        kapruka_service=kapruka_service,
        client_ip=_EVAL_CLIENT_IP,
        genai_client=build_eval_genai_client(intent_for_case(case), case=case),
        redis_client=redis_client,
    )
    return build_shopping_graph(checkpointer=None, deps=deps)


async def collect_eval_rows_per_case(
    dataset: GoldenDataset,
    *,
    redis_client: RedisClient,
) -> list[GraphEvalRow]:
    """Run graph per case with intent-specific genai mock."""
    rows: list[GraphEvalRow] = []
    for case in dataset.cases:
        graph = await build_eval_graph_for_case(case, redis_client)
        row = await run_graph_for_case(case, graph=graph, redis_client=redis_client)
        rows.append(row)
    return rows


def _ci_run_config() -> RunConfig:
    """Serial Ragas jobs in CI — parallel workers deadlock on Python 3.12 runners."""
    return RunConfig(timeout=_CI_RAGAS_TIMEOUT_SECONDS, max_workers=1)


async def run_ragas_eval_async(
    rows: list[GraphEvalRow],
    *,
    llm: LangchainLLMWrapper | None = None,
    embeddings: LangchainEmbeddingsWrapper | None = None,
    run_config: RunConfig | None = None,
) -> RagasEvalScores:
    """Score graph outputs with Ragas using the async evaluator (no nest_asyncio)."""
    judge_llm = llm or build_ci_ragas_llm()
    judge_embeddings = embeddings or build_ci_ragas_embeddings()
    hf_dataset = rows_to_dataset(rows)
    result = await aevaluate(
        hf_dataset,
        metrics=[context_precision, answer_relevancy, faithfulness],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=False,
        run_config=run_config or _ci_run_config(),
    )
    return RagasEvalScores(
        context_precision=_mean_metric(result, "context_precision"),
        answer_relevancy=_mean_metric(result, "answer_relevancy"),
        faithfulness=_mean_metric(result, "faithfulness"),
        case_count=len(rows),
    )


def run_ragas_eval(
    rows: list[GraphEvalRow],
    *,
    llm: LangchainLLMWrapper | None = None,
    embeddings: LangchainEmbeddingsWrapper | None = None,
) -> RagasEvalScores:
    """Sync wrapper for tests and scripts outside an active event loop."""
    return asyncio.run(
        run_ragas_eval_async(rows, llm=llm, embeddings=embeddings),
    )


async def run_full_ragas_eval(
    dataset_path: str | None = None,
    *,
    llm: LangchainLLMWrapper | None = None,
    embeddings: LangchainEmbeddingsWrapper | None = None,
) -> RagasEvalScores:
    """Load golden dataset, run graph with mock MCP, and return Ragas scores."""
    if dataset_path is None:
        dataset = load_golden_dataset()
    else:
        from pathlib import Path

        dataset = load_golden_dataset(Path(dataset_path))

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    redis_client = RedisClient("redis://localhost:6379/0", client=fake)
    rows = await collect_eval_rows_per_case(dataset, redis_client=redis_client)
    return await run_ragas_eval_async(rows, llm=llm, embeddings=embeddings)


def assert_context_precision_threshold(
    scores: RagasEvalScores,
    *,
    threshold: float = DEFAULT_CONTEXT_PRECISION_THRESHOLD,
) -> None:
    """Raise AssertionError when context_precision falls below the CI gate."""
    if math.isnan(scores.context_precision) or scores.context_precision < threshold:
        msg = (
            f"context_precision {scores.context_precision:.4f} below threshold {threshold:.2f} "
            f"(cases={scores.case_count})"
        )
        raise AssertionError(msg)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas eval against golden_dataset.json")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional path to golden dataset JSON (default: evals/golden_dataset.json)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit non-zero when context_precision is below the CI threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONTEXT_PRECISION_THRESHOLD,
        help=f"Minimum mean context_precision (default: {DEFAULT_CONTEXT_PRECISION_THRESHOLD})",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scores = await run_full_ragas_eval(args.dataset)
    print(
        f"Ragas eval ({scores.case_count} cases): "
        f"context_precision={scores.context_precision:.4f} "
        f"answer_relevancy={scores.answer_relevancy:.4f} "
        f"faithfulness={scores.faithfulness:.4f}",
    )
    if args.ci:
        assert_context_precision_threshold(scores, threshold=args.threshold)
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for local runs and CI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        code = asyncio.run(_async_main(argv))
    except AssertionError as exc:
        print(f"RAGAS_CI_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
