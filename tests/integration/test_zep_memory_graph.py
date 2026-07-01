"""Integration tests for Zep memory read/write in the shopping graph."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage
from tests.helpers.mock_genai import build_mock_genai_client

from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph, initial_shopping_state
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import CategoryRef, Money, ProductResult, SearchProductsOutput
from lib.zep.client import ZepClient

_TEST_API_KEY = "zep-test-api-key"
_ZEP_THREAD_ID = "zep-thread-memory-001"
_SESSION_ID = "sess-zep-memory-001"
_CLIENT_IP = "203.0.113.55"

_USER_MESSAGE = "birthday cake for mom"
_ASSISTANT_MESSAGE = "I found Chocolate Birthday Cake (LKR 4,500) for your mom's birthday."

_CHOCOLATE_CAKE = ProductResult(
    id="cake001",
    name="Chocolate Birthday Cake",
    summary="Rich chocolate birthday cake.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url="https://example.com/cake.jpg",
    category=CategoryRef(id="cat_birthday", name="Birthday", slug="birthday"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/cake",
)

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[_CHOCOLATE_CAKE],
    next_cursor=None,
    applied_filters={"q": _USER_MESSAGE, "limit": 10, "in_stock_only": False},
)


class _ZepMemoryCapture:
    """Track thread context GET and message POST calls for integration assertions."""

    def __init__(self) -> None:
        self.memory_posts: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Api-Key {_TEST_API_KEY}"

        context_path = f"/threads/{_ZEP_THREAD_ID}/context"
        messages_path = f"/threads/{_ZEP_THREAD_ID}/messages"

        if request.method == "GET" and request.url.path.endswith(context_path):
            return httpx.Response(
                200,
                json={
                    "context": ("- Customer prefers birthday gifts\n- Recipient is mom"),
                },
            )

        if request.method == "POST" and request.url.path.endswith(messages_path):
            body = json.loads(request.content)
            self.memory_posts.append(body)
            return httpx.Response(200, json={"message": "OK"})

        if request.method == "GET" and request.url.path.endswith("/threads"):
            return httpx.Response(
                200,
                json={"threads": [], "total_count": 0, "response_count": 0},
            )

        return httpx.Response(404, json={"message": "not found"})


def _mock_kapruka_service() -> Any:
    from unittest.mock import AsyncMock

    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    return mock_service


@pytest.fixture
async def zep_client_with_capture() -> tuple[ZepClient, _ZepMemoryCapture]:
    capture = _ZepMemoryCapture()
    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(capture.handler),
        base_url="https://api.getzep.com/api/v2",
    )
    client = await ZepClient.connect(_TEST_API_KEY, httpx_client=httpx_client)
    yield client, capture
    await client.close()


@pytest.mark.asyncio
async def test_shopping_graph_persists_turn_to_zep_after_chat(
    zep_client_with_capture: tuple[ZepClient, _ZepMemoryCapture],
) -> None:
    """Graph loads Zep facts, runs a turn, and POSTs user/assistant messages to Zep."""
    zep_client, capture = zep_client_with_capture
    graph = build_shopping_graph(
        deps=ShoppingGraphDeps(
            kapruka_service=_mock_kapruka_service(),
            client_ip=_CLIENT_IP,
            genai_client=build_mock_genai_client(
                search_query=_USER_MESSAGE,
                assistant_message=_ASSISTANT_MESSAGE,
            ),
            zep_client=zep_client,
        ),
    )
    state: AgentState = initial_shopping_state(
        message=_USER_MESSAGE,
        session_id=_SESSION_ID,
        zep_thread_id=_ZEP_THREAD_ID,
    )

    result = await graph.ainvoke(state)

    assert result["zep_memory_facts"] == [
        "Customer prefers birthday gifts",
        "Recipient is mom",
    ]
    assert result["assistant_message"] == _ASSISTANT_MESSAGE
    rendered = (result.get("response_html") or "") + (result.get("carousel_html") or "")
    assert "Chocolate Birthday Cake" in rendered

    assert len(capture.memory_posts) == 1
    posted_messages = capture.memory_posts[0]["messages"]
    assert len(posted_messages) == 2
    assert posted_messages[0]["role"] == "user"
    assert posted_messages[0]["content"] == _USER_MESSAGE
    assert posted_messages[1]["role"] == "assistant"
    assert posted_messages[1]["content"] == _ASSISTANT_MESSAGE


@pytest.mark.asyncio
async def test_shopping_graph_zep_node_order_includes_memory_nodes(
    zep_client_with_capture: tuple[ZepClient, _ZepMemoryCapture],
) -> None:
    """astream_events confirms load_zep_memory before analyze and write after generate."""
    zep_client, _capture = zep_client_with_capture
    graph = build_shopping_graph(
        deps=ShoppingGraphDeps(
            kapruka_service=_mock_kapruka_service(),
            client_ip=_CLIENT_IP,
            genai_client=build_mock_genai_client(
                search_query=_USER_MESSAGE,
                assistant_message=_ASSISTANT_MESSAGE,
            ),
            zep_client=zep_client,
        ),
    )
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-zep-order"}}
    state: AgentState = {
        "messages": [HumanMessage(content=_USER_MESSAGE)],
        "session_id": _SESSION_ID,
        "zep_thread_id": _ZEP_THREAD_ID,
    }

    node_names: list[str] = []
    async for event in graph.astream_events(state, config, version="v2"):
        if event.get("event") == "on_chain_start" and event.get("name") in {
            "load_zep_memory",
            "analyze_intent",
            "retrieve_hybrid_context",
            "agent_loop",
            "generate_response",
            "zep_memory_write",
        }:
            node_names.append(str(event["name"]))

    assert node_names == [
        "load_zep_memory",
        "analyze_intent",
        "retrieve_hybrid_context",
        "agent_loop",
        "generate_response",
        "zep_memory_write",
    ]
