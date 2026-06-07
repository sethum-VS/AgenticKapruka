"""End-to-end integration tests for POST /chat/stream."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import ShoppingGraphDeps
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import SearchProductsOutput
from lib.redis.client import RedisClient

_MESSAGE = "show me birthday cakes"

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[],
    next_cursor=None,
    applied_filters={"q": _MESSAGE, "limit": 10, "in_stock_only": False},
)


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


def _mock_genai_client() -> MagicMock:
    """Gemini client returning discovery intent then assistant reply."""
    mock_client = MagicMock()

    intent_response = MagicMock()
    intent_response.parsed = IntentClassification(intent="discovery")
    intent_response.text = '{"intent": "discovery"}'

    reply_response = MagicMock()
    reply_response.parsed = AssistantReply(
        message="Here are some birthday cake options from Kapruka.",
    )
    reply_response.text = reply_response.parsed.model_dump_json()

    mock_client.models.generate_content.side_effect = [intent_response, reply_response]
    return mock_client


def _mock_kapruka_service() -> AsyncMock:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    return mock_service


@pytest.fixture
def chat_flow_env(monkeypatch: pytest.MonkeyPatch) -> RedisClient:
    """App env with fakeredis and mocked Kapruka + Gemini for full graph chat stream."""
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    redis_client = RedisClient("redis://localhost:6379/0", client=fake)

    deps = ShoppingGraphDeps(
        kapruka_service=_mock_kapruka_service(),
        client_ip="127.0.0.1",
        genai_client=_mock_genai_client(),
        zep_client=None,
    )

    async def mock_build_deps(request: object, redis: RedisClient) -> ShoppingGraphDeps:
        return deps

    monkeypatch.setattr("app.routes.chat.build_shopping_graph_deps", mock_build_deps)
    monkeypatch.setattr(AsyncRedisSaver, "asetup", _fakeredis_asetup)
    return redis_client


@pytest.mark.asyncio
async def test_chat_stream_end_to_end_assistant_message(chat_flow_env: RedisClient) -> None:
    """POST /chat/stream runs shopping graph; SSE contains assistant role markup."""
    application = create_app()
    application.state.redis = chat_flow_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": _MESSAGE},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "event: message\n" in body
    assert _MESSAGE in body
    assert 'aria-label="Assistant message"' in body
    assert 'role="assistant"' in body or 'role="article"' in body
    assert "birthday cake" in body.lower()
    assert "Something went wrong" not in body
    assert 'role="alert"' not in body
