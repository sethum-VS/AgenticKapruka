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
          sse-swap="message,status"
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


def _chat_welcome_chips_harness_html() -> str:
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
      <div id="chat-messages" x-ref="messages" style="height: 200px; overflow-y: auto;">
        <div id="chat-empty-state">
          <button
            type="button"
            data-chat-suggestion="Birthday cake for mom in Colombo"
            data-testid="chat-suggestion-chip"
          >
            Birthday cake for mom in Colombo
          </button>
        </div>
      </div>
      <form
        id="chat-form"
        hx-post="/chat/stream"
        hx-ext="sse"
        sse-connect="/chat/stream"
        hx-trigger="submit"
      >
        <div
          id="chat-sse-listener"
          sse-swap="message,status"
          hx-target="#chat-messages"
          hx-swap="beforeend"
          hidden
        ></div>
        <textarea id="chat-message" x-ref="input" name="message"></textarea>
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
def test_chat_sse_done_event_clears_sending_indicator() -> None:
    """Explicit done event clears Sending… without waiting for body close."""
    sse_body = (
        "event: message\n"
        'data: <div id="user-msg">You said hello</div>\n\n'
        "event: done\n"
        "data: \n\n"
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
              return form
                && !form.classList.contains('htmx-request')
                && indicator
                && !indicator.classList.contains('htmx-request');
            }""",
            timeout=1000,
        )

        browser.close()


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
              pending.textContent = 'Searching Kapruka…';
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


@pytest.mark.browser
def test_chat_sse_clears_input_immediately_while_sending() -> None:
    """Textarea clears on submit while Sending… stays visible until the stream completes."""
    sse_body = (
        "event: message\n"
        'data: <div id="user-msg">You said hello</div>\n\n'
        "event: message\n"
        'data: <div id="assistant-final">Done</div>\n\n'
    )
    held_routes: list[Route] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        def hold_stream(route: Route) -> None:
            held_routes.append(route)

        page.route("http://localhost/chat/stream", hold_stream)
        page.set_content(_chat_sse_harness_html(include_loading_indicator=True))
        _wait_for_alpine(page)

        page.fill("#chat-message", "Hello")
        page.click('button[type="submit"]')
        page.wait_for_function(
            """() => {
              const input = document.getElementById('chat-message');
              const indicator = document.getElementById('chat-loading');
              const form = document.getElementById('chat-form');
              return input
                && input.value === ''
                && input.readOnly
                && indicator
                && indicator.classList.contains('htmx-request')
                && form
                && form.classList.contains('htmx-request');
            }""",
            timeout=1000,
        )

        assert len(held_routes) == 1
        held_routes[0].fulfill(
            status=200,
            headers={"Content-Type": "text/event-stream"},
            body=sse_body,
        )
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
def test_chat_suggestion_chip_fills_input_and_submits() -> None:
    """Clicking a welcome suggestion chip fills the input and starts the chat stream."""
    captured_bodies: list[str] = []
    sse_body = (
        "event: message\n"
        'data: <div id="user-msg">chip turn</div>\n\n'
        "event: message\n"
        'data: <div id="assistant-final">Here are cakes</div>\n\n'
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        def capture_post(route: Route) -> None:
            captured_bodies.append(route.request.post_data or "")
            route.fulfill(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                body=sse_body,
            )

        page.on(
            "request",
            lambda request: (
                captured_bodies.append(request.post_data or "")
                if request.method == "POST" and request.url.endswith("/chat/stream")
                else None
            ),
        )
        page.route("http://localhost/chat/stream", capture_post)
        page.set_content(_chat_welcome_chips_harness_html())
        _wait_for_alpine(page)

        page.click('[data-testid="chat-suggestion-chip"]')
        page.wait_for_function(
            """() => {
              const messages = document.getElementById('chat-messages');
              return messages && messages.textContent.includes('Here are cakes');
            }"""
        )

        assert page.input_value("#chat-message") == ""
        assert any("Birthday" in body and "Colombo" in body for body in captured_bodies if body)

        browser.close()


@pytest.mark.browser
def test_chat_sse_prunes_stale_carousels_after_new_carousel() -> None:
    """Only the latest product carousel remains after a follow-up search."""
    carousel_a = (
        '<div class="flex justify-start" data-role="assistant-message">'
        '<div role="assistant" aria-label="Assistant message">'
        '<div class="assistant-products" data-slot="product-carousel">'
        '<div data-testid="product-carousel" id="carousel-old">'
        '<span data-testid="product-card">Old Rs. 26,310 Cake</span>'
        "</div></div></div></div>"
    )
    carousel_b = (
        '<div class="flex justify-start" data-role="assistant-message">'
        '<div role="assistant" aria-label="Assistant message">'
        '<div class="assistant-products" data-slot="product-carousel">'
        '<div data-testid="product-carousel" id="carousel-new">'
        '<span data-testid="product-card">New Chocolate Box</span>'
        "</div></div></div></div>"
    )
    sse_body = (
        "event: message\n"
        f"data: {carousel_a}\n\n"
        "event: message\n"
        f"data: {carousel_b}\n\n"
        "event: done\n"
        "data: \n\n"
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
              const carousels = document.querySelectorAll('[data-testid="product-carousel"]');
              return carousels.length === 1 && carousels[0].id === 'carousel-new';
            }""",
            timeout=1000,
        )
        assert "26,310" not in page.inner_text("#chat-messages")

        browser.close()


@pytest.mark.browser
def test_chat_sse_done_clears_sending_without_response_html() -> None:
    """Graph completion emits done even when no assistant HTML is streamed."""
    sse_body = "event: done\n" "data: \n\n"

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
              return form
                && !form.classList.contains('htmx-request')
                && indicator
                && !indicator.classList.contains('htmx-request');
            }""",
            timeout=1000,
        )

        browser.close()
