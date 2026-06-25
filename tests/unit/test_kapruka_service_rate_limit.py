"""Unit tests for KaprukaService MCP rate-limit retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.kapruka.errors import KaprukaRateLimitError
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import SearchProductsOutput


@pytest.mark.asyncio
async def test_cached_read_retries_once_on_kapruka_rate_limit() -> None:
    """KaprukaRateLimitError triggers one backoff retry before surfacing failure."""
    redis = AsyncMock()
    mcp = MagicMock()
    service = KaprukaService(redis, mcp)

    expected = SearchProductsOutput(results=[], applied_filters={})
    calls = 0

    async def fetch() -> SearchProductsOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KaprukaRateLimitError("429", "Too many requests", retry_after_seconds=30)
        return expected

    with (
        patch("lib.kapruka.service.check_rate_limit", new_callable=AsyncMock),
        patch("lib.kapruka.service.get_cached", new_callable=AsyncMock, return_value=None),
        patch("lib.kapruka.service.set_cached", new_callable=AsyncMock),
        patch("lib.kapruka.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        result = await service._cached_read(
            client_ip="127.0.0.1",
            tool_name="kapruka_search_products",
            cache_args={"q": "cakes"},
            fetch=fetch,
            to_cache=lambda value: value.model_dump_json(),
            from_cache=lambda text: SearchProductsOutput.model_validate_json(text),
        )

    assert result == expected
    assert calls == 2
    mock_sleep.assert_awaited_once_with(5)
