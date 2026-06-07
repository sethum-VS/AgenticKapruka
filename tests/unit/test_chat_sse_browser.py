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


def _chat_sse_harness_html() -> str:
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
        <button type="submit">Send</button>
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
