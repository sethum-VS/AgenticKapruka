"""Typed wrapper for kapruka_list_categories MCP tool."""

from __future__ import annotations

import json

from lib.kapruka.errors import parse_mcp_error
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.types import ListCategoriesInput, ListCategoriesOutput

TOOL_NAME = "kapruka_list_categories"


async def list_categories(
    client: MCPHttpClient,
    *,
    depth: int = 1,
) -> ListCategoriesOutput:
    """List Kapruka category tree via MCP and return typed nested nodes."""
    categories_input = ListCategoriesInput(
        depth=depth,
        response_format="json",
    )
    params = categories_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    raw = await client.call_tool(TOOL_NAME, params)
    text = raw.strip()

    parse_mcp_error(text)
    payload = json.loads(text)
    return ListCategoriesOutput.model_validate(payload)
