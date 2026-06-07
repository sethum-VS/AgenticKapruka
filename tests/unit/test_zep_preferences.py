"""Unit tests for lib.zep.preferences cross-session extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from zep_python.types.fact import Fact
from zep_python.types.memory import Memory
from zep_python.types.session_search_response import SessionSearchResponse
from zep_python.types.session_search_result import SessionSearchResult

from lib.zep.preferences import (
    extract_preferences,
    merge_preferences_into_hybrid_context,
    parse_preferences_from_facts,
)


def test_parse_preferences_from_facts_extracts_category_currency_occasion() -> None:
    facts = [
        "User prefers birthday cakes for family gifts",
        "User shops in USD for international orders",
        "Mom's birthday is in June",
    ]

    preferences = parse_preferences_from_facts(facts)

    assert preferences["favorite_category"] == "Birthday"
    assert preferences["currency"] == "USD"
    assert preferences["past_occasion"] == "mom's birthday"


def test_parse_preferences_from_facts_empty_input() -> None:
    assert parse_preferences_from_facts([]) == {}


def test_merge_preferences_into_hybrid_context_builds_mcp_hints() -> None:
    preferences = {
        "favorite_category": "Flowers",
        "currency": "GBP",
        "past_occasion": "anniversary",
    }

    merged = merge_preferences_into_hybrid_context({}, preferences)

    assert merged["preferences"] == preferences
    assert merged["hints"]["category"] == "Flowers"
    assert merged["hints"]["currency"] == "GBP"
    assert merged["hints"]["occasion"] == "anniversary"


@pytest.mark.asyncio
async def test_extract_preferences_uses_zep_memory_search() -> None:
    search_response = SessionSearchResponse(
        results=[
            SessionSearchResult(
                fact=Fact(fact="User prefers Birthday cakes"),
                message=None,
                score=0.92,
                session_id="thread-pref-001",
                summary=None,
            ),
        ],
    )
    zep_client = AsyncMock()
    zep_client.search_session_memory.return_value = search_response

    preferences = await extract_preferences(zep_client, "thread-pref-001")

    assert preferences["favorite_category"] == "Birthday"
    zep_client.search_session_memory.assert_awaited_once()
    zep_client.get_memory.assert_not_called()


@pytest.mark.asyncio
async def test_extract_preferences_falls_back_to_get_memory_when_search_empty() -> None:
    zep_client = AsyncMock()
    zep_client.search_session_memory.return_value = SessionSearchResponse(results=[])
    zep_client.get_memory.return_value = Memory(
        facts=["Prefers Flowers bouquets"],
        messages=[],
        metadata={},
        relevant_facts=[],
        summary="",
    )

    preferences = await extract_preferences(zep_client, "thread-pref-002")

    assert preferences["favorite_category"] == "Flowers"
    zep_client.get_memory.assert_awaited_once_with("thread-pref-002")


@pytest.mark.asyncio
async def test_extract_preferences_returns_empty_on_total_failure() -> None:
    zep_client = AsyncMock()
    zep_client.search_session_memory.side_effect = RuntimeError("search down")
    zep_client.get_memory.side_effect = RuntimeError("memory down")

    preferences = await extract_preferences(zep_client, "thread-pref-003")

    assert preferences == {}
