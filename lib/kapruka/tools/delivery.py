"""Typed wrappers for kapruka_list_delivery_cities and kapruka_check_delivery MCP tools."""

from __future__ import annotations

import json

from lib.kapruka.errors import parse_mcp_error
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.types import (
    CheckDeliveryInput,
    CheckDeliveryOutput,
    ListDeliveryCitiesInput,
    ListDeliveryCitiesOutput,
)

LIST_CITIES_TOOL = "kapruka_list_delivery_cities"
CHECK_DELIVERY_TOOL = "kapruka_check_delivery"


async def list_delivery_cities(
    client: MCPHttpClient,
    *,
    query: str | None = None,
    limit: int = 25,
) -> list[str]:
    """List deliverable Kapruka cities via MCP; returns canonical city names."""
    cities_input = ListDeliveryCitiesInput(
        query=query,
        limit=limit,
        response_format="json",
    )
    params = cities_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    raw = await client.call_tool(LIST_CITIES_TOOL, params)
    text = raw.strip()

    parse_mcp_error(text)
    payload = json.loads(text)
    output = ListDeliveryCitiesOutput.model_validate(payload)
    return [city.name for city in output.cities]


async def check_delivery(
    client: MCPHttpClient,
    *,
    city: str,
    delivery_date: str | None = None,
    product_id: str | None = None,
) -> CheckDeliveryOutput:
    """Check Kapruka delivery availability for a city and optional date."""
    delivery_input = CheckDeliveryInput(
        city=city,
        delivery_date=delivery_date,
        product_id=product_id,
        response_format="json",
    )
    params = delivery_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    raw = await client.call_tool(CHECK_DELIVERY_TOOL, params)
    text = raw.strip()

    parse_mcp_error(text)
    payload = json.loads(text)
    return CheckDeliveryOutput.model_validate(payload)
