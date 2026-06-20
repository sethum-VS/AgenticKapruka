"""Unit tests for graphs.nodes.resolve_delivery_context."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.resolve_delivery_context import (
    resolve_delivery_context,
    route_after_resolve_delivery_context,
)
from graphs.state import AgentState
from lib.chat.city_resolution import CityResolution
from lib.kapruka.service import KaprukaService

_CLIENT_IP = "203.0.113.42"


def test_route_after_resolve_clarify_when_question_set() -> None:
    state: AgentState = {
        "messages": [],
        "session_id": "sess-resolve-route",
        "agent_clarifying_question": "Which Colombo zone?",
    }
    assert route_after_resolve_delivery_context(state) == "generate_response"


def test_route_after_resolve_product_id_fast_path() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="cake00ka002034 for Colombo")],
        "session_id": "sess-resolve-route",
        "delivery_context_ready": True,
    }
    assert route_after_resolve_delivery_context(state) == "call_mcp_tools"


def test_route_after_resolve_defaults_to_agent_loop() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="roses for Galle tomorrow")],
        "session_id": "sess-resolve-route",
        "delivery_context_ready": True,
    }
    assert route_after_resolve_delivery_context(state) == "agent_loop"


@pytest.mark.asyncio
async def test_resolve_delivery_context_skips_when_no_city_signal() -> None:
    service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="show me chocolates")],
        "session_id": "sess-resolve-skip",
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "en",
            "requires_delivery_validation": False,
            "target_city": None,
            "budget_max": None,
        },
    }
    result = await resolve_delivery_context(state, kapruka_service=service, client_ip=_CLIENT_IP)
    assert result == {"delivery_context_ready": True}
    service.list_delivery_cities.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_delivery_context_ambiguous_colombo_clarifies() -> None:
    service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="Birthday cake for mom in Colombo")],
        "session_id": "sess-resolve-ambiguous",
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "en",
            "requires_delivery_validation": True,
            "target_city": "Colombo",
            "budget_max": None,
        },
    }
    ambiguous = CityResolution(
        status="ambiguous",
        candidates=["Colombo 01", "Colombo 02", "Colombo 03"],
        customer_message="Colombo has several delivery zones. Which area?",
    )
    with patch(
        "graphs.nodes.resolve_delivery_context.resolve_delivery_city",
        new=AsyncMock(return_value=ambiguous),
    ):
        result = await resolve_delivery_context(
            state,
            kapruka_service=service,
            client_ip=_CLIENT_IP,
        )

    assert result["delivery_city_status"] == "ambiguous"
    assert result["agent_clarifying_question"] == ambiguous.customer_message
    assert result["agent_loop_exit_reason"] == "ask_user"
    assert result["delivery_context_ready"] is False


@pytest.mark.asyncio
async def test_resolve_delivery_context_resolved_sets_canonical_and_date() -> None:
    service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="roses for Galle tomorrow")],
        "session_id": "sess-resolve-galle",
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "en",
            "requires_delivery_validation": True,
            "target_city": "Galle",
            "budget_max": None,
        },
    }
    resolved = CityResolution(status="resolved", canonical="Galle")

    with (
        patch(
            "graphs.nodes.resolve_delivery_context.resolve_delivery_city",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "graphs.nodes.resolve_delivery_context.normalize_delivery_date",
            return_value="2026-06-13",
        ),
        patch("lib.utils.timezone.colombo_today", return_value=date(2026, 6, 12)),
    ):
        result = await resolve_delivery_context(
            state,
            kapruka_service=service,
            client_ip=_CLIENT_IP,
        )

    assert result["delivery_city_canonical"] == "Galle"
    assert result["delivery_date"] == "2026-06-13"
    assert result["session_delivery_city_canonical"] == "Galle"
    assert result["delivery_context_ready"] is True
    assert result.get("agent_clarifying_question") is None


@pytest.mark.asyncio
async def test_resolve_delivery_context_date_only_turn_reuses_session_city() -> None:
    """Date-only follow-up restores ephemeral city from session_delivery_city_canonical."""
    service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="tomorrow")],
        "session_id": "sess-resolve-kandy-date",
        "session_delivery_city_canonical": "Kandy",
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "en",
            "requires_delivery_validation": False,
            "target_city": None,
            "budget_max": None,
        },
    }
    resolved = CityResolution(status="resolved", canonical="Kandy")

    with (
        patch(
            "graphs.nodes.resolve_delivery_context.resolve_delivery_city",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "graphs.nodes.resolve_delivery_context.normalize_delivery_date",
            return_value="2026-06-13",
        ),
        patch("lib.utils.timezone.colombo_today", return_value=date(2026, 6, 12)),
    ):
        result = await resolve_delivery_context(
            state,
            kapruka_service=service,
            client_ip=_CLIENT_IP,
        )

    assert result["delivery_city_canonical"] == "Kandy"
    assert result["session_delivery_city_canonical"] == "Kandy"
    assert result["delivery_date"] == "2026-06-13"
    assert result["delivery_context_ready"] is True
