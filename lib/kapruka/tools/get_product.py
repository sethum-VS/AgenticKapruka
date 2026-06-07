"""Typed wrapper for kapruka_get_product MCP tool."""

from __future__ import annotations

import json

from lib.kapruka.errors import parse_mcp_error
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.types import GetProductInput, GetProductOutput

TOOL_NAME = "kapruka_get_product"


async def get_product(
    client: MCPHttpClient,
    *,
    product_id: str,
    currency: str = "LKR",
    type: str | None = None,
) -> GetProductOutput:
    """Fetch a single Kapruka product via MCP and return typed detail."""
    product_input = GetProductInput(
        product_id=product_id,
        currency=currency,
        type=type,
        response_format="json",
    )
    params = product_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    raw = await client.call_tool(TOOL_NAME, params)
    text = raw.strip()

    parse_mcp_error(text)
    payload = json.loads(text)
    return GetProductOutput.model_validate(payload)
