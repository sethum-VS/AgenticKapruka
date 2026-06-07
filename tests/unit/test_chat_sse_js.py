"""Tests for POST SSE bridge static script."""

from __future__ import annotations

from pathlib import Path

CHAT_SSE_JS = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "chat-sse.js"


def test_chat_sse_js_wires_post_stream_bridge() -> None:
    """chat-sse.js bridges POST /chat/stream into sse-swap listener swaps."""
    source = CHAT_SSE_JS.read_text()

    assert "htmx.createEventSource" in source
    assert "sse-swap" in source
    assert "sse-connect" in source
    assert 'CHAT_STREAM_PATH = "/chat/stream"' in source
    assert 'eventName = "message"' in source
    assert "htmx.swap" in source
    assert 'HX-Request": "true"' in source
    assert "htmx:afterSwap" in source
    assert "htmx:afterRequest" in source
