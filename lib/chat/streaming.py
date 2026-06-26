"""Map LangGraph astream updates to HTMX-compatible SSE HTML events."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.intent_heuristics import is_vague_gift_intent
from lib.chat.off_topic import is_impossible_catalog_request, is_off_topic_message
from lib.chat.sse import chunk_text, format_sse_event
from lib.chat.status_copy import SEARCHING_CATALOG
from lib.debug.trace import trace_error, trace_node_update, trace_turn_complete

logger = logging.getLogger(__name__)

CHAT_TURN_TIMEOUT_SECONDS = 90.0
_TIMEOUT_MESSAGE = (
    "This is taking longer than expected. Please try again with a more specific question."
)
_CART_ERROR_FALLBACK = "I couldn't add that — try naming the product."


def _skip_early_search_status(state: AgentState) -> bool:
    """Skip generic search status when the turn routes straight to a reply."""
    intent = state.get("intent")
    if intent in ("tracking", "checkout"):
        return True
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if not user_message.strip():
        return False
    if is_off_topic_message(user_message) or is_impossible_catalog_request(user_message):
        return True
    if is_vague_gift_intent(user_message):
        return True
    q = state.get("agent_clarifying_question")
    return bool(isinstance(q, str) and q.strip())


def _cart_error_message_from_state(state: dict[str, Any]) -> str | None:
    action = state.get("cart_action_result")
    if not isinstance(action, dict):
        return None
    if action.get("status") != "error":
        return None
    message = action.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return _CART_ERROR_FALLBACK


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
    partial_state: dict[str, Any] = {}

    if not _skip_early_search_status(state):
        early_status_message = SEARCHING_CATALOG
        thinking_html = _render_streaming_assistant(early_status_message, pending_id, oob=False)
        yield format_sse_event(thinking_html)
        stream_started = True
        status_html = _render_streaming_assistant(early_status_message, pending_id, oob=True)
        yield format_sse_event(status_html, event="status")

    thread_id = ""
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if isinstance(configurable, dict):
        thread_id = str(configurable.get("thread_id") or "")

    done_emitted = False
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
                    partial_state.update(node_update)
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
                    yield format_sse_event("", event="done")
                    done_emitted = True
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
        yield format_sse_event("", event="done")
        done_emitted = True
    except Exception as exc:
        trace_error("graph.astream failed", exc)
        logger.exception("chat stream failed during graph.astream")
        cart_message = _cart_error_message_from_state(partial_state)
        if cart_message:
            error_html = _render_streaming_assistant(cart_message, pending_id, oob=stream_started)
            if stream_started:
                error_html = f'<div id="{pending_id}" hx-swap-oob="delete"></div>{error_html}'
        else:
            error_html = (
                '<div class="flex justify-start">'
                '<div class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 '
                'text-sm text-red-800" role="alert">'
                "Something went wrong. Please try again.</div></div>"
            )
            if stream_started:
                error_html = f'<div id="{pending_id}" hx-swap-oob="delete"></div>{error_html}'
        yield format_sse_event(error_html)
        yield format_sse_event("", event="done")
        done_emitted = True
    finally:
        if not done_emitted:
            cleanup = (
                f'<div id="{pending_id}" hx-swap-oob="delete"></div>' if stream_started else ""
            )
            if cleanup:
                yield format_sse_event(cleanup)
            yield format_sse_event("", event="done")
