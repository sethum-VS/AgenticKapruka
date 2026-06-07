"""Unit tests for lib.zep.memory helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from zep_python.types.memory import Memory

from lib.zep.memory import (
    DEFAULT_FACT_LIMIT,
    append_session_messages,
    extract_memory_facts,
    format_memory_facts_block,
    get_session_memory_facts,
)


def test_extract_memory_facts_returns_last_n() -> None:
    memory = Memory(
        facts=["Prefers birthday cakes", "Lives in Colombo", "Mom's birthday in June"],
        messages=[],
        metadata={},
        relevant_facts=[],
        summary="",
    )

    result = extract_memory_facts(memory, limit=2)

    assert result == ["Lives in Colombo", "Mom's birthday in June"]


def test_extract_memory_facts_empty_memory() -> None:
    memory = Memory(facts=[], messages=[], metadata={}, relevant_facts=[], summary="")
    assert extract_memory_facts(memory) == []


def test_format_memory_facts_block_empty() -> None:
    assert format_memory_facts_block([]) == ""


def test_format_memory_facts_block_renders_bullets() -> None:
    block = format_memory_facts_block(["Prefers roses", "Budget under LKR 5000"])
    assert "Prior session facts" in block
    assert "- Prefers roses" in block
    assert "- Budget under LKR 5000" in block


@pytest.mark.asyncio
async def test_get_session_memory_facts_delegates_to_client() -> None:
    memory = Memory(
        facts=["Likes chocolate"],
        messages=[],
        metadata={},
        relevant_facts=[],
        summary="",
    )
    zep_client = AsyncMock()
    zep_client.get_memory.return_value = memory

    facts = await get_session_memory_facts(zep_client, "thread-abc")

    assert facts == ["Likes chocolate"]
    zep_client.get_memory.assert_awaited_once_with("thread-abc")


@pytest.mark.asyncio
async def test_get_session_memory_facts_returns_empty_on_error() -> None:
    zep_client = AsyncMock()
    zep_client.get_memory.side_effect = RuntimeError("not found")

    facts = await get_session_memory_facts(zep_client, "missing-thread")

    assert facts == []


@pytest.mark.asyncio
async def test_append_session_messages_posts_user_and_assistant() -> None:
    zep_client = AsyncMock()
    zep_client.add_messages.return_value = MagicMock()

    await append_session_messages(
        zep_client,
        "thread-xyz",
        "birthday cake for mom",
        "I found Chocolate Birthday Cake.",
    )

    zep_client.add_messages.assert_awaited_once()
    session_id, messages = zep_client.add_messages.await_args.args
    assert session_id == "thread-xyz"
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "birthday cake for mom"
    assert messages[1].role == "assistant"
    assert messages[1].content == "I found Chocolate Birthday Cake."


def test_default_fact_limit_is_ten() -> None:
    assert DEFAULT_FACT_LIMIT == 10
