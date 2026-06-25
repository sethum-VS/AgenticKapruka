"""Unit tests for LLM+MCP shipment address resolution (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.state import AgentState
from lib.chat.address_resolution import (
    ExtractedDestination,
    extract_destination_regex,
    resolve_shipment_address,
)
from lib.chat.city_resolution import CityResolution
from lib.kapruka.service import KaprukaService

_CLIENT_IP = "203.0.113.9"


def test_extract_destination_regex_colombo_zone() -> None:
    assert extract_destination_regex("deliver to Colombo 03") == "Colombo 03"


@pytest.mark.asyncio
async def test_resolve_shipment_address_llm_match_confirms_low_confidence() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="send to mom's place in Galle")],
    }
    extracted = ExtractedDestination(
        raw_text="mom's place in Galle",
        city_candidate="Galle",
        confidence="low",
    )

    with (
        patch(
            "lib.chat.address_resolution.extract_destination_llm",
            new=AsyncMock(return_value=extracted),
        ),
        patch(
            "lib.chat.address_resolution.resolve_delivery_city",
            new=AsyncMock(
                return_value=CityResolution(status="resolved", canonical="Galle"),
            ),
        ),
    ):
        updates = await resolve_shipment_address(
            state,
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
            genai_client=object(),
        )

    assert updates.get("agent_clarifying_question")
    assert "Galle" in updates["agent_clarifying_question"]


@pytest.mark.asyncio
async def test_resolve_shipment_address_confirmation_writes_session_city() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="yes, Colombo 03")],
        "delivery_city_candidates": ["Colombo 03", "Colombo 04"],
        "delivery_city_canonical": "Colombo 03",
        "delivery_city_raw": "Colombo 03",
    }

    updates = await resolve_shipment_address(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    assert updates.get("session_delivery_city_canonical") == "Colombo 03"
    assert updates.get("session_delivery_city_confirmed") is True
