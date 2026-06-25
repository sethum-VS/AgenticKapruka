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
    page.click('button[type="submit"]')
    page.wait_for_function(
        """() => {
          const form = document.getElementById('chat-form');
          const loading = document.getElementById('chat-loading');
          const formIdle = form && !form.classList.contains('htmx-request');
          const loadingIdle = !loading || (
            !loading.classList.contains('htmx-request')
            && !loading.classList.contains('chat-loading')
          );
          const noPendingStream = !document.querySelector('[id^="assistant-stream-"]');
          const assistants = document.querySelectorAll('[aria-label="Assistant message"]');
          const last = assistants.length ? assistants[assistants.length - 1] : null;
          const lastText = last ? (last.textContent || '').trim().toLowerCase() : '';
          const notSearching = lastText !== 'searching kapruka…';
          return formIdle && loadingIdle && noPendingStream && notSearching;
        }""",
        timeout=timeout_ms,
    )
    page.wait_for_selector('[aria-label="Assistant message"]', timeout=timeout_ms)


def extract_chat_messages_html(page: Page) -> str:
    """Return inner HTML of the chat message log for LLM-judge rubrics."""
    return page.locator("#chat-messages").inner_html()


def extract_last_assistant_html(page: Page) -> str:
    assistant = page.locator('[aria-label="Assistant message"]').last
    if assistant.count() == 0:
        return ""
    return assistant.inner_html()


def extract_last_assistant_text(page: Page) -> str:
    """Return visible text from the latest finalized assistant bubble."""
    assistants = page.locator('[aria-label="Assistant message"]')
    count = assistants.count()
    for index in range(count - 1, -1, -1):
        bubble = assistants.nth(index)
        bubble_id = bubble.get_attribute("id") or ""
        if bubble_id.startswith("assistant-stream-"):
            continue
        text = bubble.inner_text().strip()
        if text.lower() == "searching kapruka…":
            continue
        return text
    return ""
