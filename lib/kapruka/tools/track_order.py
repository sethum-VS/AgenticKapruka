"""Typed wrapper for kapruka_track_order MCP tool."""

from __future__ import annotations

import json

from lib.kapruka.errors import parse_mcp_error
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.types import TrackOrderInput, TrackOrderOutput

TOOL_NAME = "kapruka_track_order"


async def track_order(
    client: MCPHttpClient,
    *,
    order_number: str,
) -> TrackOrderOutput:
    """Look up Kapruka order tracking by post-payment order number (not order_ref)."""
    track_input = TrackOrderInput(
        order_number=order_number,
        response_format="json",
    )
    params = track_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    raw = await client.call_tool(TOOL_NAME, params)
    text = raw.strip()

    parse_mcp_error(text)
    payload = json.loads(text)
    return TrackOrderOutput.model_validate(payload)
