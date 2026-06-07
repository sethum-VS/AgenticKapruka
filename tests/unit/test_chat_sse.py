"""Unit tests for SSE formatting and chat stream helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

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
async def test_iter_chat_sse_events_yields_user_then_assistant_chunks() -> None:
    """Graph astream updates are mapped to SSE HTML events."""
    assistant_html = render_assistant_html("Hello from Kapruka assistant.")
    mock_graph = MagicMock()

    async def fake_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | None = None,
    ) -> Any:
        yield {
            "generate_response": {
                "response_html": assistant_html,
                "assistant_message": "Hello from Kapruka assistant.",
            },
        }

    mock_graph.astream = fake_astream

    events: list[str] = []
    async for event in iter_chat_sse_events(
        graph=mock_graph,
        state={},
        config={"configurable": {"thread_id": "t-1"}},
        user_html='<div id="user">hi</div>',
    ):
        events.append(event)

    assert len(events) >= 3
    assert 'data: <div id="user">hi</div>' in events[0]
    assert any("assistant-stream-" in event for event in events[1:])
    assert "Hello from Kapruka assistant." in events[-1]
    assert 'aria-label="Assistant message"' in events[-1]
    assert 'hx-swap-oob="delete"' in events[-1]


@pytest.mark.asyncio
async def test_iter_chat_sse_events_emits_error_event_on_graph_failure() -> None:
    """Unhandled graph errors yield a user-visible SSE error fragment."""
    mock_graph = MagicMock()

    async def failing_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | None = None,
    ) -> Any:
        if False:
            yield {}
        raise RuntimeError("graph exploded")

    mock_graph.astream = failing_astream

    collected: list[str] = []
    with pytest.raises(RuntimeError, match="graph exploded"):
        async for event in iter_chat_sse_events(
            graph=mock_graph,
            state={},
            config={"configurable": {"thread_id": "t-err"}},
            user_html="<p>user</p>",
        ):
            collected.append(event)

    assert collected
    assert "Something went wrong" in collected[-1]
