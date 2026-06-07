"""Map LangGraph astream updates to HTMX-compatible SSE HTML events."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from graphs.state import AgentState
from lib.chat.sse import chunk_text, format_sse_event

logger = logging.getLogger(__name__)


def _render_streaming_assistant(message: str, element_id: str, *, oob: bool) -> str:
    """Render a partial assistant bubble that can be replaced via OOB swap."""
    oob_attr = ' hx-swap-oob="outerHTML"' if oob else ""
    escaped = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<div id="{element_id}" class="flex justify-start"{oob_attr}>'
        f'<div class="max-w-[85%] rounded-2xl rounded-bl-md border border-commerce-parchment '
        f'bg-white px-4 py-3 text-sm leading-relaxed text-commerce-ink shadow-sm" '
        f'role="assistant" aria-label="Assistant message">'
        f'<p class="whitespace-pre-wrap">{escaped}</p>'
        f"</div></div>"
    )


async def iter_chat_sse_events(
    *,
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    state: AgentState,
    config: RunnableConfig,
    user_html: str,
    stream_id: str | None = None,
) -> AsyncIterator[str]:
    """Yield SSE-encoded HTML fragments while the shopping graph runs."""
    yield format_sse_event(user_html)

    pending_id = f"assistant-stream-{stream_id or secrets.token_hex(4)}"
    stream_started = False

    try:
        async for update in graph.astream(state, config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_update in update.items():
                if node_name != "generate_response":
                    continue
                response_html = node_update.get("response_html")
                assistant_message = (node_update.get("assistant_message") or "").strip()
                if not response_html:
                    continue

                text_chunks = chunk_text(assistant_message)
                if not text_chunks:
                    text_chunks = [assistant_message]

                accumulated = ""
                for piece in text_chunks:
                    accumulated = f"{accumulated} {piece}".strip() if accumulated else piece
                    html = _render_streaming_assistant(
                        accumulated,
                        pending_id,
                        oob=stream_started,
                    )
                    stream_started = True
                    yield format_sse_event(html)

                cleanup = f'<div id="{pending_id}" hx-swap-oob="delete"></div>'
                yield format_sse_event(cleanup + response_html)
    except Exception:
        logger.exception("chat stream failed during graph.astream")
        error_html = (
            '<div class="flex justify-start">'
            '<div class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 '
            'text-sm text-red-800" role="alert">'
            "Something went wrong. Please try again.</div></div>"
        )
        if stream_started:
            error_html = f'<div id="{pending_id}" hx-swap-oob="delete"></div>{error_html}'
        yield format_sse_event(error_html)
        raise
