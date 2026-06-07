"""Unit tests for Kapruka MCP HTTP JSON-RPC client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lib.kapruka.mcp_client import MCPHttpClient

_TEST_MCP_URL = "https://mcp.kapruka.com/mcp"
_TEST_SESSION_ID = "test-mcp-session"


def _jsonrpc_result(req_id: int | str, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _mcp_handler(
    request: httpx.Request,
    *,
    tool_call_attempts: dict[str, int] | None = None,
    fail_tool_calls: int = 0,
    captured_tool_calls: list[dict[str, Any]] | None = None,
) -> httpx.Response:
    if request.method == "DELETE":
        return httpx.Response(204)

    if request.method == "GET":
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="",
        )

    body = json.loads(request.content) if request.content else {}
    method = body.get("method")
    req_id = body.get("id")

    if method == "initialize":
        return httpx.Response(
            200,
            json=_jsonrpc_result(
                req_id,
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "kapruka", "version": "1.0"},
                },
            ),
            headers={
                "mcp-session-id": _TEST_SESSION_ID,
                "content-type": "application/json",
            },
        )

    if method == "notifications/initialized":
        return httpx.Response(202)

    if method == "tools/list":
        return httpx.Response(
            200,
            json=_jsonrpc_result(req_id, {"tools": []}),
            headers={"content-type": "application/json"},
        )

    if method == "tools/call":
        if captured_tool_calls is not None:
            captured_tool_calls.append(body)

        if tool_call_attempts is not None:
            tool_call_attempts["count"] = tool_call_attempts.get("count", 0) + 1
            if tool_call_attempts["count"] <= fail_tool_calls:
                return httpx.Response(503, json={"error": "service unavailable"})

        return httpx.Response(
            200,
            json=_jsonrpc_result(
                req_id,
                {
                    "content": [{"type": "text", "text": '{"results": [{"id": "cake1"}]}'}],
                    "isError": False,
                },
            ),
            headers={"content-type": "application/json"},
        )

    return httpx.Response(404, json={"message": f"unhandled method: {method}"})


@pytest.fixture
def captured_tool_calls() -> list[dict[str, Any]]:
    return []


@pytest.fixture
async def mcp_client(captured_tool_calls: list[dict[str, Any]]) -> MCPHttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return _mcp_handler(request, captured_tool_calls=captured_tool_calls)

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://mcp.kapruka.com",
    )
    client = await MCPHttpClient.connect(_TEST_MCP_URL, httpx_client=httpx_client)
    yield client
    await client.close()


async def test_call_tool_sends_jsonrpc_envelope(
    mcp_client: MCPHttpClient,
    captured_tool_calls: list[dict[str, Any]],
) -> None:
    """call_tool POSTs a tools/call JSON-RPC request with Kapruka params wrapper."""
    result = await mcp_client.call_tool(
        "kapruka_search_products",
        {"q": "birthday cake", "response_format": "json"},
    )

    assert len(captured_tool_calls) == 1
    envelope = captured_tool_calls[0]
    assert envelope["jsonrpc"] == "2.0"
    assert envelope["method"] == "tools/call"
    assert envelope["params"]["name"] == "kapruka_search_products"
    assert envelope["params"]["arguments"] == {
        "params": {"q": "birthday cake", "response_format": "json"},
    }
    assert isinstance(envelope["id"], int)
    assert result == '{"results": [{"id": "cake1"}]}'


async def test_call_tool_retries_on_5xx_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient 503 responses are retried up to MAX_RETRY_ATTEMPTS."""
    tool_call_attempts: dict[str, int] = {}
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("lib.kapruka.mcp_client.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return _mcp_handler(
            request,
            tool_call_attempts=tool_call_attempts,
            fail_tool_calls=2,
        )

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://mcp.kapruka.com",
    )
    client = await MCPHttpClient.connect(_TEST_MCP_URL, httpx_client=httpx_client)

    try:
        result = await client.call_tool("kapruka_search_products", {"q": "cake"})
    finally:
        await client.close()

    assert result == '{"results": [{"id": "cake1"}]}'
    assert tool_call_attempts["count"] == 3
    assert sleeps == [0.5, 1.0]


async def test_call_tool_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent 5xx failures exhaust retries and raise HTTPStatusError."""

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("lib.kapruka.mcp_client.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return _mcp_handler(request, tool_call_attempts={"count": 0}, fail_tool_calls=10)

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://mcp.kapruka.com",
    )
    client = await MCPHttpClient.connect(_TEST_MCP_URL, httpx_client=httpx_client)

    try:
        with pytest.raises(ExceptionGroup) as exc_info:
            await client.call_tool("kapruka_search_products", {"q": "cake"})
        assert any(
            isinstance(err, httpx.HTTPStatusError) and err.response.status_code == 503
            for err in exc_info.value.exceptions
        )
    finally:
        await client.close()


async def test_call_tool_does_not_retry_4xx() -> None:
    """Client errors (4xx) are not retried."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method in {"DELETE", "GET"}:
            return _mcp_handler(request)

        body = json.loads(request.content) if request.content else {}
        if body.get("method") == "tools/call":
            attempts["count"] += 1
            return httpx.Response(400, json={"error": "bad request"})
        return _mcp_handler(request)

    httpx_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://mcp.kapruka.com",
    )
    client = await MCPHttpClient.connect(_TEST_MCP_URL, httpx_client=httpx_client)

    try:
        with pytest.raises(ExceptionGroup):
            await client.call_tool("kapruka_search_products", {"q": "cake"})
    finally:
        await client.close()

    assert attempts["count"] == 1


async def test_close_is_idempotent(mcp_client: MCPHttpClient) -> None:
    """close() can be called multiple times without error."""
    await mcp_client.close()
    await mcp_client.close()
