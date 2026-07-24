"""Playwright helpers for HybridRAG E2E: SSE wait, DOM extraction, MCP log."""

from __future__ import annotations

from playwright.sync_api import Page


def wait_for_alpine(page: Page) -> None:
    page.wait_for_function("() => Boolean(window.Alpine)")


def reset_mcp_log(page: Page, base_url: str) -> None:
    page.request.post(f"{base_url}/e2e/mcp-calls/reset")


def reset_e2e_session(page: Page, base_url: str) -> None:
    """Clear MCP log, LangGraph checkpoints, mock planner state, and session cookie."""
    page.request.post(f"{base_url}/e2e/reset")
    page.context.clear_cookies()
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)


def fetch_mcp_tools(page: Page, base_url: str) -> list[str]:
    """Return mock MCP tool names when the E2E app exposes /e2e/mcp-calls."""
    response = page.request.get(f"{base_url}/e2e/mcp-calls")
    if response.status != 200:
        return []
    payload = response.json()
    tools = payload.get("tools", [])
    return list(tools) if isinstance(tools, list) else []


def send_chat_message(page: Page, message: str, *, timeout_ms: int = 60_000) -> None:
    """Type a message, submit, and wait until SSE streaming completes."""
    page.fill("#chat-message", message)
    completion_js = """({users, assistants}) => {
          const STATUS = new Set([
            'searching kapruka…',
            'searching our catalog…',
            'checking delivery options…',
            'curating options for your budget…',
            'putting together recommendations…',
            'thinking…',
            'sending…',
          ]);
          const curUsers = document.querySelectorAll('[data-role="user-message"]').length;
          const assistantEls = document.querySelectorAll('[aria-label="Assistant message"]');
          const curAssistants = assistantEls.length;
          const started = curUsers > users || curAssistants > assistants;
          const form = document.getElementById('chat-form');
          const formIdle = form && !form.classList.contains('htmx-request');
          const loading = document.getElementById('chat-loading');
          const loadingIdle = !loading || (
            !loading.classList.contains('htmx-request')
            && !loading.classList.contains('chat-loading')
          );
          const noPendingStream = !document.querySelector('[id^="assistant-stream-"]');
          const messages = document.getElementById('chat-messages');
          let lastUser = null;
          if (messages) {
            const kids = messages.children;
            for (let i = kids.length - 1; i >= 0; i--) {
              if (kids[i].matches('[data-role="user-message"]')) {
                lastUser = kids[i];
                break;
              }
            }
          }
          let replyText = '';
          if (lastUser) {
            let el = lastUser.nextElementSibling;
            while (el) {
              const bubble = el.matches('[aria-label="Assistant message"]')
                ? el
                : el.querySelector('[aria-label="Assistant message"]');
              if (bubble) {
                const prose = bubble.querySelector('.prose-assistant');
                replyText = ((prose && prose.textContent) || bubble.textContent || '').trim();
                break;
              }
              el = el.nextElementSibling;
            }
          }
          const lowered = replyText.toLowerCase();
          const isStatus = !lowered || STATUS.has(lowered);
          const hasFinalReply = Boolean(lastUser) && !isStatus && noPendingStream;
          return started && formIdle && loadingIdle && hasFinalReply;
        }"""
    last_error: Exception | None = None
    for attempt in range(2):
        prior_user_turns = page.locator('[data-role="user-message"]').count()
        prior_assistant_turns = page.locator('[aria-label="Assistant message"]').count()
        page.locator("#chat-form").evaluate("form => form.requestSubmit()")
        try:
            page.wait_for_function(
                completion_js,
                arg={"users": prior_user_turns, "assistants": prior_assistant_turns},
                timeout=timeout_ms,
            )
            page.wait_for_selector('[aria-label="Assistant message"]', timeout=timeout_ms)
            return
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                page.wait_for_timeout(500)
    if last_error is not None:
        raise last_error


def extract_chat_messages_html(page: Page) -> str:
    """Return inner HTML of the chat message log for LLM-judge rubrics."""
    return page.locator("#chat-messages").inner_html()


def start_new_session(
    page: Page,
    *,
    keep_cart: bool = True,
    base_url: str | None = None,
) -> None:
    """Open the New Session modal, confirm keep/clear, and wait for the reset.

    The New Session control shows a modal (``new-session-modal``); the reset only
    fires after the Keep or Clear button is clicked. Tests that click the trigger
    without dismissing the modal leave the backdrop intercepting later clicks.

    When ``base_url`` is provided, reload ``/chat`` afterward so the SSE listener and
    composer are clean (``/chat/new`` returns OOB fragments, not a full page).
    """
    page.click('[data-testid="new-chat-button"]')
    page.wait_for_selector(
        '[data-testid="new-session-modal"]',
        state="visible",
        timeout=10_000,
    )
    testid = "new-session-keep" if keep_cart else "new-session-clear"
    page.click(f'[data-testid="{testid}"]')
    page.wait_for_function(
        """() => {
          const modal = document.getElementById('new-session-modal');
          const empty = document.getElementById('chat-empty-state');
          const carousels = document.querySelectorAll('[data-testid="product-carousel"]');
          const modalHidden = !modal || modal.hidden;
          return modalHidden && empty && carousels.length === 0;
        }""",
        timeout=15_000,
    )
    if base_url:
        page.goto(f"{base_url}/chat")
        wait_for_alpine(page)


def extract_last_assistant_html(page: Page) -> str:
    assistant = page.locator('[aria-label="Assistant message"]').last
    if assistant.count() == 0:
        return ""
    return assistant.inner_html()


def extract_last_assistant_text(page: Page) -> str:
    """Return prose from the assistant bubble that follows the latest user message."""
    return page.evaluate(
        """() => {
          const messages = document.getElementById('chat-messages');
          if (!messages) return '';
          const kids = messages.children;
          let lastUser = null;
          for (let i = kids.length - 1; i >= 0; i--) {
            if (kids[i].matches('[data-role="user-message"]')) {
              lastUser = kids[i];
              break;
            }
          }
          const readBubble = (bubble) => {
            const id = bubble.id || '';
            if (id.startsWith('assistant-stream-')) return null;
            const prose = bubble.querySelector('.prose-assistant');
            const text = ((prose && prose.textContent) || bubble.textContent || '').trim();
            if (text.toLowerCase() === 'searching kapruka…') return null;
            return text;
          };
          if (lastUser) {
            let el = lastUser.nextElementSibling;
            while (el) {
              const bubble = el.matches('[aria-label="Assistant message"]')
                ? el
                : el.querySelector('[aria-label="Assistant message"]');
              if (bubble) {
                const text = readBubble(bubble);
                if (text !== null) return text;
              }
              el = el.nextElementSibling;
            }
          }
          const assistants = document.querySelectorAll('[aria-label="Assistant message"]');
          for (let i = assistants.length - 1; i >= 0; i--) {
            const text = readBubble(assistants[i]);
            if (text !== null) return text;
          }
          return '';
        }"""
    )