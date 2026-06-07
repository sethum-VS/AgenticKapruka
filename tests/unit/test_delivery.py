"""Unit tests for kapruka_list_delivery_cities and kapruka_check_delivery wrappers."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lib.kapruka.errors import KaprukaValidationError
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.delivery import (
    CHECK_DELIVERY_TOOL,
    LIST_CITIES_TOOL,
    check_delivery,
    list_delivery_cities,
)

_CITIES_JSON: dict[str, Any] = {
    "cities": [
        {"name": "Colombo 03", "aliases": ["Colombo 3", "Kollupitiya"]},
        {"name": "Colombo 07", "aliases": ["Cinnamon Gardens"]},
        {"name": "Galle", "aliases": []},
    ],
    "total_matched": 3,
    "showing": 3,
}

_DELIVERY_AVAILABLE_JSON: dict[str, Any] = {
    "city": "Colombo 03",
    "now": "2026-06-07T10:30:00+05:30",
    "checked_date": "2026-06-08",
    "available": True,
    "rate": 350.0,
    "currency": "LKR",
    "reason": None,
    "next_available_date": None,
    "perishable_warning": None,
}

_DELIVERY_UNAVAILABLE_JSON: dict[str, Any] = {
    "city": "Colombo 03",
    "now": "2026-06-07T10:30:00+05:30",
    "checked_date": "2026-06-08",
    "available": False,
    "rate": 350.0,
    "currency": "LKR",
    "reason": "Sunday delivery not available for this city.",
    "next_available_date": "2026-06-09",
    "perishable_warning": None,
}


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_CITIES_JSON))
    return client


async def test_list_delivery_cities_returns_city_names(mcp_client: MCPHttpClient) -> None:
    """Mocked MCP JSON is reduced to a list of canonical city name strings."""
    result = await list_delivery_cities(mcp_client, query="colombo", limit=10)

    assert result == ["Colombo 03", "Colombo 07", "Galle"]


async def test_list_delivery_cities_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    await list_delivery_cities(mcp_client, query="colombo", limit=10)

    mcp_client.call_tool.assert_awaited_once_with(  # type: ignore[attr-defined]
        LIST_CITIES_TOOL,
        {
            "query": "colombo",
            "limit": 10,
            "response_format": "json",
        },
    )


async def test_list_delivery_cities_validates_limit_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """ListDeliveryCitiesInput rejects limit outside 1–50."""
    with pytest.raises(ValueError, match="less than or equal to 50"):
        await list_delivery_cities(mcp_client, limit=51)

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_check_delivery_parses_available_response(
    mcp_client: MCPHttpClient,
) -> None:
    """Mocked MCP JSON is parsed; available=True when delivery is feasible."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value=json.dumps(_DELIVERY_AVAILABLE_JSON)
    )

    result = await check_delivery(
        mcp_client,
        city="Colombo 03",
        delivery_date="2026-06-08",
    )

    assert result.available is True
    assert result.city == "Colombo 03"
    assert result.checked_date == "2026-06-08"
    assert result.rate == 350.0
    assert result.currency == "LKR"
    assert result.reason is None


async def test_check_delivery_parses_unavailable_response(
    mcp_client: MCPHttpClient,
) -> None:
    """Unavailable delivery sets available=False with reason and next date."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value=json.dumps(_DELIVERY_UNAVAILABLE_JSON)
    )

    result = await check_delivery(
        mcp_client,
        city="Colombo 03",
        delivery_date="2026-06-08",
    )

    assert result.available is False
    assert result.reason == "Sunday delivery not available for this city."
    assert result.next_available_date == "2026-06-09"


async def test_check_delivery_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value=json.dumps(_DELIVERY_AVAILABLE_JSON)
    )

    await check_delivery(
        mcp_client,
        city="Colombo 03",
        delivery_date="2026-06-08",
        product_id="cake00ka002034",
    )

    mcp_client.call_tool.assert_awaited_once_with(  # type: ignore[attr-defined]
        CHECK_DELIVERY_TOOL,
        {
            "city": "Colombo 03",
            "delivery_date": "2026-06-08",
            "product_id": "cake00ka002034",
            "response_format": "json",
        },
    )


async def test_check_delivery_validates_date_format_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """CheckDeliveryInput rejects delivery_date that is not YYYY-MM-DD."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        await check_delivery(mcp_client, city="Colombo 03", delivery_date="08-06-2026")

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_check_delivery_raises_city_not_deliverable(
    mcp_client: MCPHttpClient,
) -> None:
    """city_not_deliverable MCP errors raise KaprukaValidationError."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value="Error (city_not_deliverable): City is not in the Kapruka delivery network"
    )

    with pytest.raises(KaprukaValidationError) as exc_info:
        await check_delivery(mcp_client, city="Unknown City")

    assert exc_info.value.code == "city_not_deliverable"


async def test_check_delivery_raises_date_not_deliverable(
    mcp_client: MCPHttpClient,
) -> None:
    """date_not_deliverable MCP errors raise KaprukaValidationError."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value="Error (date_not_deliverable): Selected date cannot be fulfilled"
    )

    with pytest.raises(KaprukaValidationError) as exc_info:
        await check_delivery(
            mcp_client,
            city="Colombo 03",
            delivery_date="2026-06-08",
        )

    assert exc_info.value.code == "date_not_deliverable"
