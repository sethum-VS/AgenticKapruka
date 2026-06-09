"""Typed wrapper for kapruka_search_products MCP tool."""

from __future__ import annotations

import json
import re

from lib.kapruka.errors import parse_mcp_error
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.types import SearchProductsInput, SearchProductsOutput

TOOL_NAME = "kapruka_search_products"
_NO_PRODUCTS = re.compile(r"^No products found for ", re.IGNORECASE)


async def search_products(
    client: MCPHttpClient,
    *,
    q: str,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
    sort: str = "relevance",
    limit: int = 10,
    cursor: str | None = None,
    currency: str = "LKR",
) -> SearchProductsOutput:
    """Search Kapruka catalog via MCP and return typed results."""
    search_input = SearchProductsInput(
        q=q,
        category=category,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
        sort=sort,
        limit=limit,
        cursor=cursor,
        currency=currency,
        response_format="json",
    )
    params = search_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    raw = await client.call_tool(TOOL_NAME, params)
    text = raw.strip()
    if _NO_PRODUCTS.match(text):
        return SearchProductsOutput(
            results=[],
            next_cursor=None,
            applied_filters={
                key: value
                for key, value in params.items()
                if key in {"q", "category", "sort", "limit", "in_stock_only"}
            },
        )

    parse_mcp_error(text)
    payload = json.loads(text)
    return SearchProductsOutput.model_validate(payload)
