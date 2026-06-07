"""Unit tests for kapruka_list_categories wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.list_categories import TOOL_NAME, list_categories

_SAMPLE_JSON: dict[str, Any] = {
    "categories": [
        {
            "name": "Cakes",
            "url": "https://www.kapruka.com/online/cakes",
            "children": [
                {
                    "name": "Birthday",
                    "url": "https://www.kapruka.com/online/cakes/birthday",
                    "children": [],
                },
                {
                    "name": "Wedding",
                    "url": "https://www.kapruka.com/online/cakes/wedding",
                    "children": [],
                },
            ],
        },
        {
            "name": "Flowers",
            "url": "https://www.kapruka.com/online/flowers",
            "children": [],
        },
    ],
}


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_SAMPLE_JSON))
    return client


async def test_list_categories_parses_nested_tree(mcp_client: MCPHttpClient) -> None:
    """Mocked MCP JSON is parsed into a nested category tree."""
    result = await list_categories(mcp_client, depth=2)

    assert len(result.categories) == 2
    cakes = result.categories[0]
    assert cakes.name == "Cakes"
    assert cakes.url == "https://www.kapruka.com/online/cakes"
    assert len(cakes.children) == 2
    assert cakes.children[0].name == "Birthday"
    assert cakes.children[1].name == "Wedding"
    assert cakes.children[0].children == []

    flowers = result.categories[1]
    assert flowers.name == "Flowers"
    assert flowers.children == []


async def test_list_categories_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    await list_categories(mcp_client, depth=1)

    mcp_client.call_tool.assert_awaited_once_with(  # type: ignore[attr-defined]
        TOOL_NAME,
        {
            "depth": 1,
            "response_format": "json",
        },
    )


async def test_list_categories_validates_depth_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """ListCategoriesInput rejects depth outside 1–2."""
    with pytest.raises(ValueError, match="less than or equal to 2"):
        await list_categories(mcp_client, depth=3)

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]
