"""Integration tests for budget-only refinement across multi-turn chocolate search."""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from google.genai import types
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry

from graphs.nodes.agent_loop import AgentPlannerStep
from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import (
    ShoppingGraphDeps,
    append_message_state,
    build_shopping_graph,
    initial_shopping_state,
)
from lib.chat.city_resolution import CityResolution
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.types import (
    CategoryRef,
    CheckDeliveryOutput,
    Money,
    ProductResult,
    SearchProductsOutput,
)
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient

_THREAD_ID = "thread-budget-refine-001"
_SESSION_ID = "sess-budget-refine-001"
_CLIENT_IP = "203.0.113.55"


def _rendered_html(state: dict[str, Any]) -> str:
    """Assistant bubble plus OOB carousel fragment (SSE split delivery)."""
    return (state.get("response_html") or "") + (state.get("carousel_html") or "")


_CHOCOLATE_PRODUCT = ProductResult(
    id="choc001",
    name="Heart Chocolate Gift Box",
    summary="Assorted milk chocolates.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url="https://example.com/choc.jpg",
    category=CategoryRef(id="cat_choc", name="Chocolate", slug="chocolate"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/choc",
)

_GREETING_CARD = ProductResult(
    id="card001",
    name="Greeting Card",
    summary="Generic greeting card.",
    price=Money(amount=1200.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url=None,
    category=CategoryRef(id="cat_cards", name="Greeting Cards", slug="greeting-cards"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/card",
)


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
async def checkpointer(redis_client: RedisClient) -> AsyncRedisSaver:
    with patch.object(AsyncRedisSaver, "asetup", _fakeredis_asetup):
        return await get_checkpointer(redis_client)


def _discovery_mock_genai() -> MagicMock:
    mock_client = MagicMock()

    def generate_content(
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        _ = model, contents, kwargs
        response = MagicMock()
        if config is not None and config.response_schema is IntentClassification:
            response.parsed = IntentClassification(intent="discovery")
            response.text = json.dumps({"intent": "discovery"})
            return response
        if config is not None and config.response_schema is AssistantReply:
            response.parsed = AssistantReply(message="Here are chocolate gifts within your budget.")
            response.text = json.dumps({"message": "Here are chocolate gifts within your budget."})
            return response
        response.parsed = IntentClassification(intent="discovery")
        response.text = json.dumps({"intent": "discovery"})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client


@pytest.mark.asyncio
async def test_clarify_chocolate_budget_multiturn_keeps_chocolate_carousel(
    checkpointer: AsyncRedisSaver,
) -> None:
    """QA Scenario 1 turns 1–3: chocolate thread survives budget-only refinement."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.side_effect = [
        SearchProductsOutput(
            results=[_CHOCOLATE_PRODUCT],
            next_cursor=None,
            applied_filters={"q": "chocolate gift", "limit": 10, "in_stock_only": False},
        ),
        SearchProductsOutput(
            results=[_GREETING_CARD],
            next_cursor=None,
            applied_filters={"q": "chocolate gift", "max_price": 6000.0},
        ),
    ]

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_discovery_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": _THREAD_ID}}

    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "chocolate gift"},
            rationale="search chocolate",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "gift voucher"},
            rationale="should not run on budget turn",
        ),
    ]

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        side_effect=planner_steps,
    ) as mock_plan:
        turn1 = await graph.ainvoke(
            initial_shopping_state(
                message="Something with chocolate for my wife",
                session_id=_SESSION_ID,
                thread_id=_THREAD_ID,
            ),
            config,
        )
        assert turn1.get("session_product_focus") == "chocolate"
        assert turn1.get("last_search_products")

        turn2 = await graph.ainvoke(
            append_message_state("Keep it under 6000 rupees."),
            config,
        )

    assert turn2.get("session_budget_max") == 6000.0
    html = _rendered_html(turn2)
    assert "Heart Chocolate Gift Box" in html
    assert "Greeting Card" not in html
    assert turn2.get("last_visible_products")
    assert mock_plan.call_count <= 1


@pytest.mark.asyncio
async def test_budget_refinement_carousel_stable_after_delivery_turn(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Turn 3 delivery answer keeps budget-refined chocolate carousel visible."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.side_effect = [
        SearchProductsOutput(
            results=[_CHOCOLATE_PRODUCT],
            next_cursor=None,
            applied_filters={"q": "chocolate gift", "limit": 10, "in_stock_only": False},
        ),
        SearchProductsOutput(
            results=[_GREETING_CARD],
            next_cursor=None,
            applied_filters={"q": "chocolate gift", "max_price": 6000.0},
        ),
    ]
    mock_service.check_delivery.return_value = CheckDeliveryOutput(
        city="Kandy",
        now="2026-06-25T10:00:00+05:30",
        checked_date="2026-06-28",
        available=True,
        rate=500.0,
        currency="LKR",
        reason=None,
        next_available_date=None,
        perishable_warning=None,
    )
    mock_service.list_delivery_cities = AsyncMock(return_value=["Kandy"])

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_discovery_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-budget-delivery-001"}}

    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "chocolate gift"},
            rationale="search chocolate",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
        AgentPlannerStep(action="finish", rationale="budget turn should not replan search"),
        AgentPlannerStep(
            action="call_tool",
            tool_name=CHECK_DELIVERY_TOOL,
            tool_args={"city": "Kandy", "delivery_date": "2026-06-28"},
            rationale="delivery check",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
    ]

    with (
        patch(
            "graphs.nodes.agent_loop._plan_next_step_sync",
            side_effect=planner_steps,
        ),
        patch(
            "graphs.nodes.resolve_delivery_context.resolve_delivery_city",
            new=AsyncMock(return_value=CityResolution(status="resolved", canonical="Kandy")),
        ),
        patch("lib.utils.timezone.colombo_today", return_value=date(2026, 6, 25)),
    ):
        await graph.ainvoke(
            initial_shopping_state(
                message="Something with chocolate for my wife",
                session_id="sess-budget-delivery-001",
                thread_id="thread-budget-delivery-001",
            ),
            config,
        )
        await graph.ainvoke(append_message_state("Keep it under 6000 rupees."), config)
        turn3 = await graph.ainvoke(
            append_message_state("can you deliver to Kandy this Sunday?"),
            config,
        )

    html = _rendered_html(turn3)
    assert "Heart Chocolate Gift Box" in html
    assert "Greeting Card" not in html


@pytest.mark.asyncio
async def test_budget_refinement_filters_snack_noise_from_mock_mcp(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Budget turn drops curry powder and snack bars when MCP returns cheap noise first."""
    snack = ProductResult(
        id="snack001",
        name="Chocolate Snack Bar",
        summary="Mini bar.",
        price=Money(amount=70.0, currency="LKR"),
        compare_at_price=None,
        in_stock=True,
        stock_level="high",
        image_url=None,
        category=CategoryRef(id="cat_snack", name="Snacks", slug="snacks"),
        rating=None,
        ships_internationally=False,
        url="https://www.kapruka.com/snack",
    )
    curry = ProductResult(
        id="curry001",
        name="Ruhunu Curry Powder",
        summary="Spice pack.",
        price=Money(amount=350.0, currency="LKR"),
        compare_at_price=None,
        in_stock=True,
        stock_level="high",
        image_url=None,
        category=CategoryRef(id="cat_grocery", name="Grocery", slug="grocery"),
        rating=None,
        ships_internationally=False,
        url="https://www.kapruka.com/curry",
    )
    cake = ProductResult(
        id="cake001",
        name="Say Cheers Chocolate Cake",
        summary="Birthday cake.",
        price=Money(amount=3660.0, currency="LKR"),
        compare_at_price=None,
        in_stock=True,
        stock_level="high",
        image_url=None,
        category=CategoryRef(id="cat_birthday", name="Birthday", slug="birthday"),
        rating=None,
        ships_internationally=False,
        url="https://www.kapruka.com/cake",
    )
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.side_effect = [
        SearchProductsOutput(
            results=[_CHOCOLATE_PRODUCT],
            next_cursor=None,
            applied_filters={"q": "chocolate gift"},
        ),
        SearchProductsOutput(
            results=[snack, curry, cake],
            next_cursor=None,
            applied_filters={"q": "birthday chocolate cake", "max_price": 6000.0},
        ),
    ]
    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_discovery_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-budget-noise-001"}}

    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "chocolate gift"},
            rationale="search chocolate",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
        AgentPlannerStep(action="finish", rationale="budget turn should not replan"),
    ]

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        side_effect=planner_steps,
    ):
        await graph.ainvoke(
            initial_shopping_state(
                message="Something with chocolate for my wife",
                session_id="sess-budget-noise-001",
                thread_id="thread-budget-noise-001",
            ),
            config,
        )
        turn2 = await graph.ainvoke(
            append_message_state("Keep it under 6000 rupees."),
            config,
        )

    html = _rendered_html(turn2)
    carousel_lower = html.lower()
    assert "curry" not in carousel_lower
    assert "snack" not in carousel_lower
    assert "cake" in carousel_lower or "cheers" in carousel_lower
