"""Unit tests for KaprukaService transient MCP failure retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lib.kapruka.errors import KaprukaError, KaprukaNotFoundError
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import (
    CategoryRef,
    GetProductOutput,
    Money,
    ProductAttributes,
    ProductShipping,
    SearchProductsOutput,
)


@pytest.mark.asyncio
async def test_cached_read_retries_once_on_transient_kapruka_error() -> None:
    """Generic KaprukaError triggers one backoff retry before surfacing failure."""
    redis = AsyncMock()
    mcp = MagicMock()
    service = KaprukaService(redis, mcp)

    expected = SearchProductsOutput(results=[], applied_filters={})
    calls = 0

    async def fetch() -> SearchProductsOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KaprukaError("upstream_error", "Kapruka service unavailable")
        return expected

    with (
        patch("lib.kapruka.service.check_rate_limit", new_callable=AsyncMock),
        patch("lib.kapruka.service.get_cached", new_callable=AsyncMock, return_value=None),
        patch("lib.kapruka.service.set_cached", new_callable=AsyncMock),
        patch("lib.kapruka.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        result = await service._cached_read(
            client_ip="127.0.0.1",
            tool_name="kapruka_get_product",
            cache_args={"product_id": "cake00ka002034"},
            fetch=fetch,
            to_cache=lambda value: value.model_dump_json(),
            from_cache=lambda text: SearchProductsOutput.model_validate_json(text),
        )

    assert result == expected
    assert calls == 2
    mock_sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_cached_read_retries_once_on_http_5xx() -> None:
    """HTTP 5xx from MCP transport triggers one service-level retry."""
    redis = AsyncMock()
    mcp = MagicMock()
    service = KaprukaService(redis, mcp)

    expected = GetProductOutput(
        id="cake00ka002034",
        name="Cake",
        description="",
        summary="",
        price=Money(amount=1000.0, currency="LKR"),
        compare_at_price=None,
        in_stock=True,
        stock_level="high",
        category=CategoryRef(id="cat", name="Cakes", slug="cakes"),
        variants=[],
        images=[],
        attributes=ProductAttributes(),
        shipping=ProductShipping(
            ships_from="Colombo",
            ships_internationally=False,
            restricted_countries=[],
        ),
        rating=None,
        url="https://example.com",
    )
    calls = 0
    request = httpx.Request("POST", "https://mcp.kapruka.com/mcp")
    response = httpx.Response(503, request=request)

    async def fetch() -> GetProductOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.HTTPStatusError("503", request=request, response=response)
        return expected

    with (
        patch("lib.kapruka.service.check_rate_limit", new_callable=AsyncMock),
        patch("lib.kapruka.service.get_cached", new_callable=AsyncMock, return_value=None),
        patch("lib.kapruka.service.set_cached", new_callable=AsyncMock),
        patch("lib.kapruka.service.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await service._cached_read(
            client_ip="127.0.0.1",
            tool_name="kapruka_get_product",
            cache_args={"product_id": "cake00ka002034"},
            fetch=fetch,
            to_cache=lambda value: value.model_dump_json(),
            from_cache=lambda text: GetProductOutput.model_validate_json(text),
        )

    assert result == expected
    assert calls == 2


@pytest.mark.asyncio
async def test_cached_read_does_not_retry_not_found() -> None:
    """KaprukaNotFoundError is not retried."""
    redis = AsyncMock()
    mcp = MagicMock()
    service = KaprukaService(redis, mcp)

    async def fetch() -> SearchProductsOutput:
        raise KaprukaNotFoundError("product_not_found", "No such product")

    with (
        patch("lib.kapruka.service.check_rate_limit", new_callable=AsyncMock),
        patch("lib.kapruka.service.get_cached", new_callable=AsyncMock, return_value=None),
        patch("lib.kapruka.service.set_cached", new_callable=AsyncMock),
        patch("lib.kapruka.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        pytest.raises(KaprukaNotFoundError),
    ):
        await service._cached_read(
            client_ip="127.0.0.1",
            tool_name="kapruka_get_product",
            cache_args={"product_id": "missing"},
            fetch=fetch,
            to_cache=lambda value: value.model_dump_json(),
            from_cache=lambda text: SearchProductsOutput.model_validate_json(text),
        )

    mock_sleep.assert_not_awaited()
