"""Ragas evaluation runner for the shopping graph against the golden dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
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
from graphs.nodes.analyze_intent import PROCEED_CHECKOUT_MESSAGE, IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph, initial_shopping_state
from graphs.state import AgentState, Intent
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.redis.cart import add_item
from lib.redis.client import RedisClient

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


def _apply_situational_flavor(message: str) -> str:
    """Prepend Sri Lankan empathy markers for shadow-test local_flavor gate."""
    lowered = message.lower()
    if any(marker in lowered for marker in ("aiyo", "machan", "hodata")):
        return message
    return f"{_SITUATIONAL_FLAVOR_PREFIX}{message}"


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
                return "I found these Kapruka options: " + ", ".join(lines) + "."

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


def build_eval_genai_client(intent: Intent | None = None) -> MagicMock:
    """Gemini client mock: structured intent then faithful catalog reply.

    When ``intent`` is None, infer routing from the user message (E2E / shadow tests).
    """
    client = MagicMock()
    default_intent: Intent = intent or "discovery"
    intent_response = MagicMock()
    intent_response.parsed = IntentClassification(intent=default_intent)
    intent_response.text = json.dumps({"intent": default_intent})

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

        if config is not None and config.response_schema is AssistantReply:
            message = _synthesize_assistant_reply(contents)
            if _is_concierge_system_instruction(config):
                message = _apply_situational_flavor(message)
            response.parsed = AssistantReply(message=message)
            response.text = json.dumps({"message": message})
            return response

        response.parsed = intent_response.parsed
        response.text = intent_response.text
        return response

    client.models.generate_content.side_effect = generate_content
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
    tool_results = result.get("tool_results")
    tool_dict = tool_results if isinstance(tool_results, dict) else {}

    return GraphEvalRow(
        user_input=case.user_query,
        response=_plain_response_from_state(result),
        retrieved_contexts=contexts_from_tool_results(tool_dict),
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
        genai_client=build_eval_genai_client(intent_for_case(case)),
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
