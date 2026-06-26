"""Tests for Alpine chatHelpers static script."""

from __future__ import annotations

from pathlib import Path

CHAT_HELPERS_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "chat-helpers.js"
)


def test_chat_helpers_js_registers_alpine_component() -> None:
    """chat-helpers.js defines chatHelpers with scroll and focus helpers."""
    source = CHAT_HELPERS_JS.read_text()

    assert 'Alpine.data("chatHelpers"' in source
    assert "scrollToBottom" in source
    assert "focusInput" in source
    assert "htmx:afterSwap" in source
    assert "htmx:afterRequest" in source
    assert 'target?.id === "chat-messages"' in source
    assert 'elt?.id === "chat-form"' in source


def test_chat_helpers_js_wires_enter_key_submit() -> None:
    """Enter in the chat textarea submits; Shift+Enter is left for new lines."""
    source = CHAT_HELPERS_JS.read_text()

    assert 'event.key !== "Enter"' in source
    assert "event.shiftKey" in source
    assert "event.preventDefault()" in source
    assert 'form.classList.contains("htmx-request")' in source
    assert "form.requestSubmit()" in source


def test_chat_helpers_js_wires_suggestion_chip_click() -> None:
    """data-chat-suggestion chips fill the chat input and submit the form."""
    source = CHAT_HELPERS_JS.read_text()

    assert "[data-chat-suggestion]" in source
    assert 'getAttribute("data-chat-suggestion")' in source
    assert 'querySelector("#chat-message")' in source
    assert "form.requestSubmit()" in source
    assert "input.value = suggestion" in source
