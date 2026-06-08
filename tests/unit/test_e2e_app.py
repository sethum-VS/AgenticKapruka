"""Contract tests for the E2E app factory (no browser required)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from tests.e2e.e2e_app import create_e2e_app, get_e2e_mcp_client

from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL


@pytest.fixture
def e2e_app():
    return create_e2e_app()


@pytest.mark.asyncio
async def test_e2e_app_serves_chat_and_mcp_debug_route(e2e_app) -> None:
    async with e2e_app.router.lifespan_context(e2e_app):
        transport = ASGITransport(app=e2e_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            chat = await client.get("/chat")
            mcp_calls = await client.get("/e2e/mcp-calls")

    assert chat.status_code == 200
    assert "Kapruka Gift Assistant" in chat.text
    assert mcp_calls.status_code == 200
    assert mcp_calls.json() == {"tools": []}


@pytest.mark.asyncio
async def test_e2e_mcp_mock_records_search_without_create_order(e2e_app) -> None:
    mcp = get_e2e_mcp_client()
    await mcp.call_tool(SEARCH_PRODUCTS_TOOL, {"q": "birthday cake"})
    assert SEARCH_PRODUCTS_TOOL in mcp.call_log
    assert CREATE_ORDER_TOOL not in mcp.call_log
