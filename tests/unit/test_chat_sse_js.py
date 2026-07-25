"""Tests for POST SSE bridge static script."""

from __future__ import annotations

from pathlib import Path

CHAT_SSE_JS = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "chat-sse.js"
NEW_SESSION_JS = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "new-session.js"


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
    assert 'event.eventName === "carousel"' in source
    assert "swapCarouselHtml" in source
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
    assert "clearTimeout(statusFlushTimer)" in source
    assert "statusFlushTimer = null" in source
    assert "statusShownAt = 0" in source
    assert "hideLoadingIndicator" in source
    assert "showLoadingIndicator" in source
    assert "indicator.hidden = true" in source
    assert 'indicator.setAttribute("aria-hidden", "true")' in source
    assert '!form.classList.contains("htmx-request")' in source


def test_chat_sse_js_hides_loading_indicator_on_deactivate() -> None:
    """Deactivate clears loading text and hides the indicator instead of resetting Sending…."""
    source = CHAT_SSE_JS.read_text()

    assert 'span.textContent = ""' in source
    assert "indicator.hidden = false" in source


def test_chat_sse_js_uses_abort_controller_with_backend_timeout_buffer() -> None:
    source = CHAT_SSE_JS.read_text()

    assert "AbortController" in source
    assert "CHAT_STREAM_TIMEOUT_BUFFER_MS = 10_000" in source
    assert "CHAT_STREAM_TIMEOUT_DEFAULT_MS = 130_000" in source
    assert "getChatStreamTimeoutMs" in source
    assert 'dataset?.chatTimeoutMs' in source
    assert 'chatStreamAbortReason = "timeout"' in source
    assert "controller.abort()" in source
    assert "signal: controller.signal" in source


def test_chat_sse_js_exposes_abort_for_new_session() -> None:
    source = CHAT_SSE_JS.read_text()

    assert "let chatStreamController = null" in source
    assert "window.abortChatStream = abortChatStream" in source
    assert "window.abortActiveChatStream = abortChatStream" in source
    assert 'reason === "new-session"' in source
    assert "abortChatStream" in source


def test_chat_sse_js_handles_timeout_by_clearing_pending_bubbles() -> None:
    """Client timeout clears stuck Searching bubbles and shows a retry notice."""
    source = CHAT_SSE_JS.read_text(encoding="utf-8")

    assert "showStreamTimeoutMessage" in source
    assert "chat-stream-timeout" in source
    assert 'reason === "timeout"' in source
    assert "STREAM_TROUBLE_MESSAGE" in source
    assert "That took too long" in source
    assert "clearOrphanedPendingBubblesWithNotice" in source
    timeout_block_start = source.index('if (reason === "timeout")')
    timeout_block = source[timeout_block_start : timeout_block_start + 350]
    assert "removePendingAssistantBubbles();" in timeout_block
    assert "showStreamTimeoutMessage();" in timeout_block


def test_chat_sse_js_removes_pending_bubble_on_stream_error() -> None:
    """Stream errors remove assistant-stream-* bubbles (mirrors server error path)."""
    source = CHAT_SSE_JS.read_text()

    assert "removePendingAssistantBubbles" in source
    assert '[id^="assistant-stream-"]' in source
    assert "successful: false" in source


def test_chat_sse_js_updates_loading_text_from_status_events() -> None:
    """Status SSE payloads update #chat-loading span text and aria-label."""
    source = CHAT_SSE_JS.read_text()

    assert "updateLoadingStatusText" in source
    assert 'data-testid="chat-loading-text"' in source
    assert "parseStatusTextFromHtml" in source
    assert "DEFAULT_LOADING_TEXT" in source


def test_new_session_js_prompts_before_post() -> None:
    source = NEW_SESSION_JS.read_text()

    assert 'data-testid="new-chat-button"' in source
    assert "new-session-modal" in source
    assert "keep_cart=" in source
    assert 'keepCart ? "true" : "false"' in source
    assert "abortChatStream" in source
    assert "chat-empty-state-template" in source
    assert "htmx-request" in source
