"""Browser verification for incremental chat SSE swaps."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, Route, sync_playwright

CHAT_SSE_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "chat-sse.js"
).read_text()
CHAT_HELPERS_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "chat-helpers.js"
).read_text()


def _chat_sse_harness_html(*, include_loading_indicator: bool = False) -> str:
    loading_block = ""
    if include_loading_indicator:
        loading_block = """
        <div
          id="chat-loading"
          class="htmx-indicator"
          aria-label="Sending message"
        >
          <span>Sending…</span>
        </div>"""
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <base href="http://localhost/" />
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/htmx-ext-sse@2.2.2"></script>
    <script>{CHAT_SSE_JS}</script>
    <script>{CHAT_HELPERS_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body hx-ext="sse">
    <div x-data="chatHelpers()">
      <div id="chat-messages" x-ref="messages" style="height: 200px; overflow-y: auto;"></div>
      <form
        id="chat-form"
        hx-post="/chat/stream"
        hx-ext="sse"
        sse-connect="/chat/stream"
        hx-trigger="submit"
      >
        <div
          id="chat-sse-listener"
          sse-swap="message"
          hx-target="#chat-messages"
          hx-swap="beforeend"
          hidden
        ></div>
        <textarea id="chat-message" x-ref="input" name="message">Hello</textarea>
        <button type="submit">Send</button>{loading_block}
      </form>
    </div>
  </body>
</html>"""


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function(
        "() => window.Alpine && document.querySelector('[x-data]')?._x_dataStack"
    )


@pytest.mark.browser
def test_chat_sse_streams_incremental_assistant_tokens() -> None:
    """Submitting the chat form appends each SSE message event into #chat-messages."""
    sse_body = (
        "event: message\n"
        'data: <div id="user-msg">You said hello</div>\n\n'
        "event: message\n"
        'data: <div id="assistant-partial">Here are</div>\n\n'
        "event: message\n"
        'data: <div id="assistant-partial">Here are birthday cakes</div>\n\n'
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        def handle_stream(route: Route) -> None:
            route.fulfill(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                body=sse_body,
            )

        page.route("http://localhost/chat/stream", handle_stream)
        page.set_content(_chat_sse_harness_html())
        _wait_for_alpine(page)

        page.click('button[type="submit"]')
        page.wait_for_function(
            """() => {
              const el = document.getElementById('chat-messages');
              return el && el.textContent.includes('birthday cakes');
            }"""
        )

        text = page.inner_text("#chat-messages")
        assert "You said hello" in text
        assert "birthday cakes" in text
        assert "Here are" in text

        browser.close()


@pytest.mark.browser
def test_chat_sse_clears_sending_indicator_after_stream() -> None:
    """Sending… indicator and disabled controls clear within 1s after stream completes."""
    sse_body = (
        "event: message\n"
        'data: <div id="user-msg">You said hello</div>\n\n'
        "event: message\n"
        'data: <div id="assistant-final">Done</div>\n\n'
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        def handle_stream(route: Route) -> None:
            route.fulfill(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                body=sse_body,
            )

        page.route("http://localhost/chat/stream", handle_stream)
        page.set_content(_chat_sse_harness_html(include_loading_indicator=True))
        _wait_for_alpine(page)

        page.click('button[type="submit"]')
        page.wait_for_function(
            """() => {
              const form = document.getElementById('chat-form');
              const indicator = document.getElementById('chat-loading');
              const input = document.getElementById('chat-message');
              const button = document.querySelector('#chat-form button[type="submit"]');
              return form
                && !form.classList.contains('htmx-request')
                && indicator
                && !indicator.classList.contains('htmx-request')
                && input
                && !input.readOnly
                && button
                && !button.disabled;
            }""",
            timeout=1000,
        )

        browser.close()


@pytest.mark.browser
def test_chat_sse_clears_sending_and_pending_bubble_on_error() -> None:
    """HTTP stream failure clears Sending… and removes assistant-stream-* bubbles."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        page.route(
            "http://localhost/chat/stream",
            lambda route: route.fulfill(status=500, body="error"),
        )
        page.set_content(_chat_sse_harness_html(include_loading_indicator=True))
        _wait_for_alpine(page)

        page.evaluate(
            """() => {
              const messages = document.getElementById('chat-messages');
              const pending = document.createElement('div');
              pending.id = 'assistant-stream-deadbeef';
              pending.textContent = 'Searching catalog…';
              messages.appendChild(pending);
            }"""
        )

        page.click('button[type="submit"]')
        page.wait_for_function(
            """() => {
              const form = document.getElementById('chat-form');
              const indicator = document.getElementById('chat-loading');
              const input = document.getElementById('chat-message');
              const button = document.querySelector('#chat-form button[type="submit"]');
              const pending = document.getElementById('assistant-stream-deadbeef');
              return form
                && !form.classList.contains('htmx-request')
                && indicator
                && !indicator.classList.contains('htmx-request')
                && input
                && !input.readOnly
                && button
                && !button.disabled
                && !pending;
            }""",
            timeout=1000,
        )

        browser.close()
