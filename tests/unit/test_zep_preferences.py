"""Unit tests for lib.zep.preferences cross-session extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from zep_cloud.types.entity_edge import EntityEdge
from zep_cloud.types.graph_search_results import GraphSearchResults
from zep_cloud.types.thread_context_response import ThreadContextResponse

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

    merged = merge_preferences_into_hybrid_context(
        {},
        preferences,
        user_message="anniversary flowers",
    )

    assert merged["preferences"] == preferences
    assert merged["hints"]["category"] == "Flowers"
    assert merged["hints"]["currency"] == "GBP"
    assert merged["hints"]["occasion"] == "anniversary"


@pytest.mark.asyncio
async def test_extract_preferences_uses_zep_graph_search() -> None:
    search_response = GraphSearchResults(
        context="- User prefers Birthday cakes",
        edges=[
            EntityEdge(
                name="prefers",
                fact="User prefers Birthday cakes",
                created_at="2026-01-01T00:00:00Z",
                source_node_uuid="src-uuid",
                target_node_uuid="tgt-uuid",
                uuid_="edge-uuid",
            ),
        ],
    )
    zep_client = AsyncMock()
    zep_client.search_graph.return_value = search_response

    preferences = await extract_preferences(zep_client, "thread-pref-001")

    assert preferences["favorite_category"] == "Birthday"
    zep_client.search_graph.assert_awaited_once()
    zep_client.get_user_context.assert_not_called()


@pytest.mark.asyncio
async def test_extract_preferences_falls_back_to_user_context_when_search_empty() -> None:
    zep_client = AsyncMock()
    zep_client.search_graph.return_value = GraphSearchResults()
    zep_client.get_user_context.return_value = ThreadContextResponse(
        context="User shops in USD",
    )

    preferences = await extract_preferences(zep_client, "thread-pref-002")

    assert preferences["currency"] == "USD"
    zep_client.get_user_context.assert_awaited_once()
