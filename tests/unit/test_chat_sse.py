"""Unit tests for SSE formatting and chat stream helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from graphs.nodes.generate_response import render_assistant_html
from lib.chat.sse import chunk_text, format_sse_event
from lib.chat.streaming import iter_chat_sse_events


def test_format_sse_event_includes_event_and_data_lines() -> None:
    """SSE events use event: and data: fields terminated by a blank line."""
    encoded = format_sse_event("<div>hello</div>")
    assert encoded.startswith("event: message\n")
    assert "data: <div>hello</div>\n" in encoded
    assert encoded.endswith("\n\n")


def test_format_sse_event_status_event_type() -> None:
    """Status SSE events use event: status for OOB thinking-bubble updates."""
    encoded = format_sse_event('<div id="pending" hx-swap-oob="outerHTML">…</div>', event="status")
    assert encoded.startswith("event: status\n")
    assert 'hx-swap-oob="outerHTML"' in encoded


def test_format_sse_event_multiline_data() -> None:
    """Multiline HTML is split across multiple data: lines."""
    encoded = format_sse_event("line-one\nline-two")
    assert "data: line-one\n" in encoded
    assert "data: line-two\n" in encoded


def test_chunk_text_splits_words() -> None:
    """chunk_text groups words for progressive streaming."""
    chunks = chunk_text("one two three four five", words_per_chunk=2)
    assert chunks == ["one two", "three four", "five"]


@pytest.mark.asyncio
async def test_iter_chat_sse_events_yields_thinking_bubble_on_stream_start() -> None:
    """User turn is followed by pending bubble and early Searching our catalog status SSE."""
    mock_graph = MagicMock()

    async def empty_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | list[str] | None = None,
    ) -> Any:
        if False:
            yield ("updates", {})

    mock_graph.astream = empty_astream

    events: list[str] = []
    async for event in iter_chat_sse_events(
        graph=mock_graph,
        state={},
        config={"configurable": {"thread_id": "t-think"}},
        user_html='<div id="user">hi</div>',
        stream_id="abc123",
    ):
        events.append(event)

    assert len(events) >= 3
    assert 'data: <div id="user">hi</div>' in events[0]
    assert "Searching our catalog…" in events[1]
    assert 'id="assistant-stream-abc123"' in events[1]
    assert 'hx-swap-oob="outerHTML"' not in events[1]
    assert events[2].startswith("event: status\n")
    assert "Searching our catalog…" in events[2]
    assert 'hx-swap-oob="outerHTML"' in events[2]


@pytest.mark.asyncio
async def test_iter_chat_sse_events_maps_custom_status_to_status_sse() -> None:
    """LangGraph custom status events become SSE status events with OOB HTML."""
    mock_graph = MagicMock()

    async def status_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | list[str] | None = None,
    ) -> Any:
        yield ("custom", {"type": "status", "message": "Checking delivery…"})
        if False:
            yield ("updates", {})

    mock_graph.astream = status_astream

    events: list[str] = []
    async for event in iter_chat_sse_events(
        graph=mock_graph,
        state={},
        config={"configurable": {"thread_id": "t-status"}},
        user_html="<p>user</p>",
        stream_id="status1",
    ):
        events.append(event)

    status_events = [event for event in events if event.startswith("event: status\n")]
    assert len(status_events) == 2
    assert "Searching our catalog…" in status_events[0]
    assert "Checking delivery…" in status_events[1]
    assert 'id="assistant-stream-status1"' in status_events[0]
    assert 'hx-swap-oob="outerHTML"' in status_events[0]


@pytest.mark.asyncio
async def test_iter_chat_sse_events_yields_user_then_assistant_chunks() -> None:
    """Graph astream updates are mapped to SSE HTML events."""
    assistant_html = render_assistant_html("Hello from Kapruka assistant.")
    mock_graph = MagicMock()

    async def fake_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | list[str] | None = None,
    ) -> Any:
        yield (
            "updates",
            {
                "generate_response": {
                    "response_html": assistant_html,
                    "assistant_message": "Hello from Kapruka assistant.",
                },
            },
        )

    mock_graph.astream = fake_astream

    events: list[str] = []
    async for event in iter_chat_sse_events(
        graph=mock_graph,
        state={},
        config={"configurable": {"thread_id": "t-1"}},
        user_html='<div id="user">hi</div>',
    ):
        events.append(event)

    assert len(events) >= 5
    assert 'data: <div id="user">hi</div>' in events[0]
    assert "Searching our catalog…" in events[1]
    assert events[2].startswith("event: status\n")
    assert "Searching our catalog…" in events[2]
    assert any("assistant-stream-" in event for event in events[3:])
    assert any("Hello from Kapruka assistant." in event for event in events)
    assert any('aria-label="Assistant message"' in event for event in events)
    assert any('hx-swap-oob="delete"' in event for event in events)
    assert events[-1].startswith("event: done\n")


@pytest.mark.asyncio
async def test_iter_chat_sse_events_yields_timeout_partial_on_wall_clock_exceed() -> None:
    """Per-turn wall-clock guard yields graceful timeout copy when astream stalls."""
    mock_graph = MagicMock()

    async def slow_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | list[str] | None = None,
    ) -> Any:
        import asyncio

        await asyncio.sleep(0.05)
        if False:
            yield ("updates", {})

    mock_graph.astream = slow_astream

    collected: list[str] = []
    with patch("lib.chat.streaming.CHAT_TURN_TIMEOUT_SECONDS", 0.01):
        async for event in iter_chat_sse_events(
            graph=mock_graph,
            state={},
            config={"configurable": {"thread_id": "t-timeout"}},
            user_html="<p>user</p>",
            stream_id="timeout1",
        ):
            collected.append(event)

    assert collected
    assert any("longer than expected" in event for event in collected)
    assert sum("longer than expected" in event for event in collected) == 1


@pytest.mark.asyncio
async def test_iter_chat_sse_events_emits_error_event_on_graph_failure() -> None:
    """Unhandled graph errors yield a user-visible SSE error fragment."""
    mock_graph = MagicMock()

    async def failing_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | list[str] | None = None,
    ) -> Any:
        if False:
            yield {}
        raise RuntimeError("graph exploded")

    mock_graph.astream = failing_astream

    collected: list[str] = []
    async for event in iter_chat_sse_events(
        graph=mock_graph,
        state={},
        config={"configurable": {"thread_id": "t-err"}},
        user_html="<p>user</p>",
    ):
        collected.append(event)

    assert collected
    assert "Searching our catalog…" in collected[1]
    assert any("Something went wrong" in event for event in collected)
    assert any('hx-swap-oob="delete"' in event for event in collected)
    assert collected[-1].startswith("event: done\n")
