"""Unit tests for evals/ragas_eval.py and CI Ragas gate."""

from __future__ import annotations

import json
from pathlib import Path

import fakeredis.aioredis
import pytest
from evals.golden_dataset import GoldenCase, GoldenDataset
from evals.ragas_eval import (
    DEFAULT_CONTEXT_PRECISION_THRESHOLD,
    GraphEvalRow,
    assert_context_precision_threshold,
    build_eval_genai_client,
    build_eval_graph_for_case,
    contexts_from_tool_results,
    intent_for_case,
    run_full_ragas_eval,
    run_graph_for_case,
    run_ragas_eval_async,
)
from google.genai import types
from tests.fixtures.mcp_mock import SEARCH_PRODUCTS_JSON

from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.redis.client import RedisClient


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


def test_intent_for_case_maps_categories_to_general() -> None:
    case = GoldenCase(
        id="disc-categories",
        scenario="discovery",
        user_query="What gifts can I buy?",
        expected_tools=[LIST_CATEGORIES_TOOL],
        reference_answer="Categories list.",
    )
    assert intent_for_case(case) == "general"


def test_contexts_from_tool_results_serializes_payloads() -> None:
    contexts = contexts_from_tool_results({SEARCH_PRODUCTS_TOOL: SEARCH_PRODUCTS_JSON})
    assert len(contexts) == 1
    parsed = json.loads(contexts[0])
    assert parsed["results"][0]["name"] == "Chocolate Birthday Cake"


def test_build_eval_genai_client_returns_intent_and_reply() -> None:
    client = build_eval_genai_client("discovery")
    intent_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="birthday cake",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentClassification,
        ),
    )
    assert intent_response.parsed.intent == "discovery"

    tool_block = json.dumps({SEARCH_PRODUCTS_TOOL: SEARCH_PRODUCTS_JSON}, indent=2)
    user_prompt = (
        "Customer message:\ncake\n\n"
        "tool_results (sole source of truth for catalog facts):\n"
        f"{tool_block}"
    )
    reply_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AssistantReply,
        ),
    )
    assert "Chocolate Birthday Cake" in reply_response.parsed.message


def test_build_eval_genai_client_adds_situational_flavor_for_concierge() -> None:
    client = build_eval_genai_client("discovery")
    tool_block = json.dumps({SEARCH_PRODUCTS_TOOL: SEARCH_PRODUCTS_JSON}, indent=2)
    user_prompt = (
        "Customer message:\nI broke up and need gentle flowers\n\n"
        "tool_results (sole source of truth for catalog facts):\n"
        f"{tool_block}"
    )
    reply_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are the Kapruka gift concierge — warm, locally grounded.",
            response_mime_type="application/json",
            response_schema=AssistantReply,
        ),
    )
    message = reply_response.parsed.message.lower()
    assert "aiyo" in message
    assert "machan" in message
    assert "hodata" in message


@pytest.mark.asyncio
async def test_run_graph_for_discovery_case(redis_client: RedisClient) -> None:
    case = GoldenCase(
        id="eval-disc",
        scenario="discovery",
        user_query="Show me birthday cakes for my mom",
        expected_tools=[SEARCH_PRODUCTS_TOOL],
        reference_answer="Search birthday cakes.",
    )
    graph = await build_eval_graph_for_case(case, redis_client)
    row = await run_graph_for_case(case, graph=graph, redis_client=redis_client)
    assert "birthday" in row.user_input.lower()
    assert row.response
    assert row.retrieved_contexts
    assert "Chocolate Birthday Cake" in row.response or row.retrieved_contexts[0]


@pytest.mark.ragas
@pytest.mark.asyncio
async def test_run_ragas_eval_meets_context_precision_threshold() -> None:
    rows = [
        GraphEvalRow(
            user_input="Show me birthday cakes",
            response="I found Chocolate Birthday Cake (LKR 4,500).",
            retrieved_contexts=[json.dumps(SEARCH_PRODUCTS_JSON)],
            reference="I'll search Kapruka for birthday cakes with prices.",
        ),
    ]
    scores = await run_ragas_eval_async(rows)
    assert scores.case_count == 1
    assert scores.context_precision >= DEFAULT_CONTEXT_PRECISION_THRESHOLD


@pytest.mark.ragas
@pytest.mark.asyncio
async def test_run_full_ragas_eval_on_golden_dataset(redis_client: RedisClient) -> None:
    _ = redis_client
    scores = await run_full_ragas_eval()
    assert scores.case_count >= 20
    assert scores.context_precision >= DEFAULT_CONTEXT_PRECISION_THRESHOLD
    assert_context_precision_threshold(scores)


def test_assert_context_precision_threshold_raises_when_low() -> None:
    from evals.ragas_eval import RagasEvalScores

    low_scores = RagasEvalScores(
        context_precision=0.2,
        answer_relevancy=0.5,
        faithfulness=0.5,
        case_count=1,
    )
    with pytest.raises(AssertionError, match="below threshold"):
        assert_context_precision_threshold(low_scores, threshold=0.7)


def test_mini_golden_dataset_loader(tmp_path: Path) -> None:
    path = tmp_path / "mini.json"
    path.write_text(
        """{
  "version": "1",
  "cases": [{
    "id": "mini-001",
    "scenario": "discovery",
    "user_query": "birthday cake",
    "expected_tools": ["kapruka_search_products"],
    "reference_answer": "Search cakes."
  }]
}""",
        encoding="utf-8",
    )
    dataset = GoldenDataset.model_validate(json.loads(path.read_text()))
    assert len(dataset.cases) == 1
