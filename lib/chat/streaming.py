"""Map LangGraph astream updates to HTMX-compatible SSE HTML events."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from graphs.state import AgentState
from lib.chat.sse import chunk_text, format_sse_event
from lib.debug.trace import trace_error, trace_node_update, trace_turn_complete

logger = logging.getLogger(__name__)

CHAT_TURN_TIMEOUT_SECONDS = 90.0
_TIMEOUT_MESSAGE = (
    "This is taking longer than expected. Please try again with a more specific question."
)


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


def _normalize_astream_chunk(
    chunk: object,
) -> tuple[str, object] | None:
    """Map LangGraph astream output to (mode, payload) for updates/custom modes."""
    if isinstance(chunk, tuple) and len(chunk) == 2:
        mode, payload = chunk
        if isinstance(mode, str):
            return mode, payload
    if isinstance(chunk, dict):
        return "updates", chunk
    return None


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

    early_status_message = "Searching Kapruka…"
    thinking_html = _render_streaming_assistant(early_status_message, pending_id, oob=False)
    yield format_sse_event(thinking_html)
    stream_started = True
    status_html = _render_streaming_assistant(early_status_message, pending_id, oob=True)
    yield format_sse_event(status_html, event="status")

    thread_id = ""
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if isinstance(configurable, dict):
        thread_id = str(configurable.get("thread_id") or "")

    try:
        async with asyncio.timeout(CHAT_TURN_TIMEOUT_SECONDS):
            async for chunk in graph.astream(state, config, stream_mode=["updates", "custom"]):
                normalized = _normalize_astream_chunk(chunk)
                if normalized is None:
                    continue
                mode, payload = normalized

                if mode == "custom":
                    if isinstance(payload, dict) and payload.get("type") == "status":
                        status_message = str(payload.get("message") or "").strip()
                        if status_message:
                            status_html = _render_streaming_assistant(
                                status_message,
                                pending_id,
                                oob=True,
                            )
                            yield format_sse_event(status_html, event="status")
                    continue

                if mode != "updates" or not isinstance(payload, dict):
                    continue

                for node_name, node_update in payload.items():
                    if not isinstance(node_update, dict):
                        continue
                    trace_node_update(node_name, node_update)
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
                    trace_turn_complete(
                        thread_id=thread_id,
                        assistant_message=assistant_message,
                        response_html_chars=len(response_html or ""),
                    )
    except TimeoutError:
        trace_error("graph.astream exceeded wall-clock timeout", TimeoutError())
        logger.warning(
            "chat stream timed out after %.0fs for thread %s",
            CHAT_TURN_TIMEOUT_SECONDS,
            thread_id or "(unknown)",
        )
        timeout_html = _render_streaming_assistant(_TIMEOUT_MESSAGE, pending_id, oob=stream_started)
        if stream_started:
            timeout_html = f'<div id="{pending_id}" hx-swap-oob="delete"></div>{timeout_html}'
        yield format_sse_event(timeout_html)
    except Exception as exc:
        trace_error("graph.astream failed", exc)
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
