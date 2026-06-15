"""Unit tests for lib.chat.city_resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lib.chat.city_resolution import (
    build_city_not_found_message,
    resolve_delivery_city,
)
from lib.kapruka.service import KaprukaService

_CLIENT_IP = "203.0.113.42"
_COLOMBO_ZONES = [
    "Colombo 01",
    "Colombo 02",
    "Colombo 03",
    "Colombo 04",
    "Colombo 05",
    "Colombo 06",
]


@pytest.mark.asyncio
async def test_resolve_delivery_city_missing_raw_city() -> None:
    service = AsyncMock(spec=KaprukaService)
    resolution = await resolve_delivery_city(service, _CLIENT_IP, None)
    assert resolution.status == "missing"
    assert resolution.customer_message == "Which city should we deliver to?"
    service.list_delivery_cities.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_delivery_city_exact_zone_match() -> None:
    service = AsyncMock(spec=KaprukaService)
    service.list_delivery_cities.return_value = _COLOMBO_ZONES
    resolution = await resolve_delivery_city(service, _CLIENT_IP, "Colombo 03")
    assert resolution.status == "resolved"
    assert resolution.canonical == "Colombo 03"


@pytest.mark.asyncio
async def test_resolve_delivery_city_bare_colombo_is_ambiguous() -> None:
    service = AsyncMock(spec=KaprukaService)
    service.list_delivery_cities.return_value = _COLOMBO_ZONES
    resolution = await resolve_delivery_city(service, _CLIENT_IP, "Colombo")
    assert resolution.status == "ambiguous"
    assert resolution.candidates == _COLOMBO_ZONES[:5]
    assert resolution.customer_message is not None
    assert "Colombo 01" in resolution.customer_message
    assert "Colombo 05" in resolution.customer_message


@pytest.mark.asyncio
async def test_resolve_delivery_city_single_galle_match() -> None:
    service = AsyncMock(spec=KaprukaService)
    service.list_delivery_cities.return_value = ["Galle"]
    resolution = await resolve_delivery_city(service, _CLIENT_IP, "Galle")
    assert resolution.status == "resolved"
    assert resolution.canonical == "Galle"


@pytest.mark.asyncio
async def test_resolve_delivery_city_not_found_when_empty_results() -> None:
    service = AsyncMock(spec=KaprukaService)
    service.list_delivery_cities.return_value = []
    resolution = await resolve_delivery_city(service, _CLIENT_IP, "Atlantis")
    assert resolution.status == "not_found"
    assert resolution.customer_message == build_city_not_found_message()
