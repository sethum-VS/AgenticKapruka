"""Browser verification for Alpine chatHelpers scroll behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright

CHAT_HELPERS_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "chat-helpers.js"
).read_text()


def _chat_harness_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <script>{CHAT_HELPERS_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body>
    <div x-data="chatHelpers()" style="height: 200px; display: flex; flex-direction: column;">
      <div
        id="chat-messages"
        x-ref="messages"
        style="flex: 1; min-height: 0; overflow-y: auto; border: 1px solid #ccc;"
      >
        <div style="height: 120px;">msg 1</div>
        <div style="height: 120px;">msg 2</div>
        <div style="height: 120px;">msg 3</div>
        <div style="height: 120px;">msg 4</div>
      </div>
      <textarea id="chat-message" x-ref="input"></textarea>
    </div>
  </body>
</html>"""


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function(
        "() => window.Alpine && document.querySelector('[x-data]')?._x_dataStack"
    )


@pytest.mark.browser
def test_chat_helpers_scrolls_to_bottom_on_htmx_after_swap() -> None:
    """htmx:afterSwap on #chat-messages triggers scrollToBottom in the browser."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content(_chat_harness_html())
        _wait_for_alpine(page)

        before = page.evaluate(
            """() => {
              const el = document.getElementById('chat-messages');
              return {
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
              };
            }"""
        )
        assert before["scrollTop"] < before["scrollHeight"] - before["clientHeight"]

        page.evaluate(
            """() => {
              const target = document.getElementById('chat-messages');
              const bubble = document.createElement('div');
              bubble.style.height = '120px';
              bubble.textContent = 'new message';
              target.appendChild(bubble);
              document.body.dispatchEvent(
                new CustomEvent('htmx:afterSwap', { detail: { target } })
              );
            }"""
        )

        after = page.evaluate(
            """() => {
              const el = document.getElementById('chat-messages');
              return {
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
              };
            }"""
        )
        assert after["scrollTop"] > before["scrollTop"]
        assert after["scrollTop"] + after["clientHeight"] >= after["scrollHeight"]

        browser.close()


@pytest.mark.browser
def test_chat_helpers_focuses_input_after_successful_request() -> None:
    """htmx:afterRequest on #chat-form refocuses the message textarea."""
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <script>{CHAT_HELPERS_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body>
    <div x-data="chatHelpers()">
      <div id="chat-messages" x-ref="messages" style="height: 100px; overflow-y: auto;"></div>
      <form id="chat-form">
        <textarea id="chat-message" x-ref="input"></textarea>
      </form>
    </div>
  </body>
</html>"""

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content(html)
        _wait_for_alpine(page)
        page.evaluate("() => document.getElementById('chat-message').blur()")

        page.evaluate(
            """() => {
              const form = document.getElementById('chat-form');
              document.body.dispatchEvent(
                new CustomEvent('htmx:afterRequest', {
                  detail: { elt: form, successful: true },
                })
              );
            }"""
        )

        focused = page.evaluate("() => document.activeElement?.id === 'chat-message'")
        assert focused is True

        browser.close()
