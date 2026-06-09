"""Unit tests for async Zep Cloud client wrapper."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lib.zep.client import ZepClient

_TEST_API_KEY = "zep-test-api-key"


def _mock_zep_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["Authorization"] == f"Api-Key {_TEST_API_KEY}"

    if request.method == "GET" and request.url.path.endswith("/threads"):
        return httpx.Response(
            200,
            json={"threads": [], "total_count": 0, "response_count": 0},
        )

    if request.method == "POST" and request.url.path.endswith("/users"):
        body = json.loads(request.content)
        return httpx.Response(201, json={"user_id": body["user_id"]})

    if request.method == "POST" and request.url.path.endswith("/threads"):
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "thread_id": body["thread_id"],
                "user_id": body["user_id"],
            },
        )

    return httpx.Response(404, json={"message": "not found"})


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    return httpx.MockTransport(_mock_zep_handler)


@pytest.fixture
async def zep_client(mock_transport: httpx.MockTransport) -> ZepClient:
    httpx_client = httpx.AsyncClient(
        transport=mock_transport,
        base_url="https://api.getzep.com/api/v2",
    )
    client = await ZepClient.connect(_TEST_API_KEY, httpx_client=httpx_client)
    yield client
    await client.close()


async def test_zep_client_initializes_with_api_key() -> None:
    """connect() builds AsyncZep with ZEP_API_KEY sent as Api-Key authorization."""
    captured: dict[str, Any] = {}

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        return _mock_zep_handler(request)

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(capturing_handler),
        base_url="https://api.getzep.com/api/v2",
    )

    client = await ZepClient.connect(_TEST_API_KEY, httpx_client=httpx_client)
    await client.health_check()

    assert captured["authorization"] == f"Api-Key {_TEST_API_KEY}"
    assert client.sdk._client_wrapper.api_key == _TEST_API_KEY

    await client.close()


async def test_zep_client_health_check_with_mock_http(zep_client: ZepClient) -> None:
    """health_check returns True when list_threads succeeds."""
    assert await zep_client.health_check() is True


async def test_zep_client_health_check_returns_false_on_auth_failure(
    mock_transport: httpx.MockTransport,
) -> None:
    """health_check returns False when Zep rejects the API key."""

    def unauthorized_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={"message": "invalid api key"})

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(unauthorized_handler),
        base_url="https://api.getzep.com/api/v2",
    )
    client = await ZepClient.connect(_TEST_API_KEY, httpx_client=httpx_client)

    assert await client.health_check() is False

    await client.close()


async def test_zep_client_list_threads(zep_client: ZepClient) -> None:
    """list_threads returns ThreadListResponse from mocked HTTP."""
    result = await zep_client.list_threads(page_number=1, page_size=10)

    assert result.threads == []
    assert result.total_count == 0


async def test_zep_client_create_session(zep_client: ZepClient) -> None:
    """create_session creates a user and thread in Zep."""
    thread = await zep_client.create_session(
        "thread-abc123",
        user_id="user-456",
    )

    assert thread.thread_id == "thread-abc123"
    assert thread.user_id == "user-456"


async def test_zep_client_close_is_idempotent(zep_client: ZepClient) -> None:
    """close() can be called multiple times without error."""
    await zep_client.close()
    await zep_client.close()
