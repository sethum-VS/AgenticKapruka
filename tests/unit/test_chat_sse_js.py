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
    assert "swapStatusHtml" in source
    assert 'event.eventName === "status"' in source
    assert "htmx.swap" in source
    assert 'HX-Request": "true"' in source
    assert "htmx:afterSwap" in source
    assert "htmx:afterRequest" in source
    assert 'form.classList.contains("htmx-request")' in source
    assert "submitButton.disabled = true" in source
    assert 'messageInput.value = ""' in source
    assert "new FormData(form)" in source


def test_chat_sse_js_clears_loading_on_success_and_error() -> None:
    """Loading state clears via finally and htmx:afterRequest backup on both paths."""
    source = CHAT_SSE_JS.read_text()

    assert "toggleRequestState(form, false)" in source
    assert "finally" in source
    assert "registerAfterRequestBackup" in source
    assert 'document.addEventListener("htmx:afterRequest"' in source
    assert "elt.id !== CHAT_FORM_ID" in source
    assert "submitButton.disabled = false" in source
    assert "messageInput.readOnly = false" in source
    assert 'indicator?.classList.remove("htmx-request", "chat-loading")' in source


def test_chat_sse_js_uses_abort_controller_timeout() -> None:
    source = CHAT_SSE_JS.read_text()

    assert "AbortController" in source
    assert "CHAT_STREAM_TIMEOUT_MS = 90_000" in source
    assert "controller.abort()" in source
    assert "signal: controller.signal" in source


def test_chat_sse_js_removes_pending_bubble_on_stream_error() -> None:
    """Stream errors remove assistant-stream-* bubbles (mirrors server error path)."""
    source = CHAT_SSE_JS.read_text()

    assert "removePendingAssistantBubbles" in source
    assert '[id^="assistant-stream-"]' in source
    assert "removePendingAssistantBubbles();" in source
    assert "successful: false" in source


def test_chat_sse_js_updates_loading_text_from_status_events() -> None:
    """Status SSE payloads update #chat-loading span text and aria-label."""
    source = CHAT_SSE_JS.read_text()

    assert "updateLoadingStatusText" in source
    assert 'data-testid="chat-loading-text"' in source
    assert "parseStatusTextFromHtml" in source
    assert 'DEFAULT_LOADING_TEXT = "Sending…"' in source
