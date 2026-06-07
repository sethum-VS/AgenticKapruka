"""Unit tests for Zep session create and resume."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import fakeredis.aioredis
import httpx
import pytest

from lib.redis.client import RedisClient
from lib.zep.client import ZepClient
from lib.zep.session import SESSION_TTL_SECONDS, get_or_create_session, session_mapping_key

_TEST_API_KEY = "zep-test-api-key"
_TEST_SESSION_ID = "browser-sess-abc123"


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
async def zep_client_with_counter() -> AsyncIterator[tuple[ZepClient, dict[str, int]]]:
    counter = {"create_count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/sessions"):
            counter["create_count"] += 1
            body = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "session_id": body["session_id"],
                    "user_id": body.get("user_id"),
                    "metadata": body.get("metadata"),
                },
            )
        return httpx.Response(404, json={"message": "not found"})

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.getzep.com/api/v2",
    )
    client = await ZepClient.connect(_TEST_API_KEY, httpx_client=httpx_client)
    yield client, counter
    await client.close()


async def test_get_or_create_session_creates_zep_thread_on_first_visit(
    redis_client: RedisClient,
    zep_client_with_counter: tuple[ZepClient, dict[str, int]],
) -> None:
    """First visit creates a Zep memory session and stores the mapping in Redis."""
    zep_client, counter = zep_client_with_counter

    zep_thread_id = await get_or_create_session(redis_client, zep_client, _TEST_SESSION_ID)

    assert zep_thread_id == _TEST_SESSION_ID
    assert counter["create_count"] == 1

    key = session_mapping_key(_TEST_SESSION_ID)
    stored = await redis_client.client.get(key)
    assert stored == _TEST_SESSION_ID

    ttl = await redis_client.client.ttl(key)
    assert 0 < ttl <= SESSION_TTL_SECONDS


async def test_get_or_create_session_resumes_existing_mapping(
    redis_client: RedisClient,
    zep_client_with_counter: tuple[ZepClient, dict[str, int]],
) -> None:
    """Second visit with the same session_id resumes without creating a new Zep thread."""
    zep_client, counter = zep_client_with_counter

    first = await get_or_create_session(redis_client, zep_client, _TEST_SESSION_ID)
    second = await get_or_create_session(redis_client, zep_client, _TEST_SESSION_ID)

    assert first == second == _TEST_SESSION_ID
    assert counter["create_count"] == 1
