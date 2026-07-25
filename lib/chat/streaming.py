"""Map LangGraph astream updates to HTMX-compatible SSE HTML events."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.intent_heuristics import is_vague_gift_intent
from lib.chat.off_topic import is_impossible_catalog_request, is_off_topic_message
from lib.chat.sse import chunk_text, format_sse_event
from lib.chat.status_copy import SEARCHING_CATALOG, THINKING
from lib.debug.trace import trace_error, trace_node_update, trace_turn_complete
from lib.genai.completions import turn_deadline
from lib.genai.errors import is_rate_limited, is_transient_nim_error

logger = logging.getLogger(__name__)

_TIMEOUT_MESSAGE = (
    "That took too long — please try again. Your last results are still above "
    "if you want to refine the budget or pick another gift."
)
_TIMEOUT_PARTIAL_MESSAGE = (
    "Here are options that match what I found so far — "
    "I'm still polishing the reply. Feel free to ask for refinements."
)
_TIMEOUT_BUDGET_PARTIAL_MESSAGE = (
    "Here are options within your budget — "
    "I'm still polishing the reply. Feel free to ask for refinements."
)
_CART_ERROR_FALLBACK = "I couldn't add that — try naming the product."
_RATE_LIMIT_MESSAGE = (
    "I'm getting a lot of requests right now — give me a moment and try that again."
)


def chat_turn_timeout_seconds() -> float:
    """Return configured wall-clock timeout for a single chat SSE turn."""
    return float(get_settings().chat_turn_timeout_seconds)


def _partial_timeout_payload(
    partial_state: dict[str, Any],
    *,
    initial_state: AgentState | None = None,
) -> tuple[str, str | None]:
    """Build timeout reply text and optional carousel HTML from partial graph state.

    Returns ``(message_or_html, carousel_oob_or_none)``. When no products are
    available, returns the generic timeout message with no carousel.
    """
    from graphs.nodes.generate_response import (
        build_products_carousel_html,
        render_assistant_html,
        render_carousel_oob_html,
    )
    from lib.chat.intent_heuristics import is_budget_refinement_message
    from lib.chat.product_curation import refine_last_search_by_budget

    merged: dict[str, Any] = {}
    if initial_state:
        merged.update(dict(initial_state))
    merged.update(partial_state)

    products = list(merged.get("last_search_products") or [])
    if not products:
        products = list(merged.get("last_visible_products") or [])
    budget_max = merged.get("session_budget_max")
    currency = str(merged.get("currency") or "LKR")
    user_message = _extract_latest_user_message(merged.get("messages") or [])
    budget_applied = False
    if isinstance(budget_max, (int, float)) and budget_max > 0 and products:
        from lib.chat.product_curation import product_price_amount

        refined = refine_last_search_by_budget(
            products,
            budget_max=float(budget_max),
            currency=currency,
            session_product_focus=(
                merged.get("session_product_focus")
                if isinstance(merged.get("session_product_focus"), str)
                else None
            ),
            session_search_query=(
                merged.get("session_search_query")
                if isinstance(merged.get("session_search_query"), str)
                else None
            ),
            session_recipient_hint=(
                merged.get("session_recipient_hint")
                if isinstance(merged.get("session_recipient_hint"), str)
                else None
            ),
            user_message=user_message,
            hybrid_context=merged.get("hybrid_context")
            if isinstance(merged.get("hybrid_context"), dict)
            else None,
        )
        if refined:
            products = refined
            budget_applied = True
        else:
            # Safety net: simple price filter when focus refine yields nothing.
            simple = [
                item
                for item in products
                if isinstance(item, dict)
                and (price := product_price_amount(item)) is not None
                and price <= float(budget_max)
            ]
            if simple:
                products = simple
                budget_applied = True
            elif is_budget_refinement_message(user_message):
                # Budget turn with only over-budget prior picks — do not resurface them.
                return _TIMEOUT_MESSAGE, None

    if not products:
        return _TIMEOUT_MESSAGE, None

    tool_results = {
        "kapruka_search_products": {"results": products},
    }
    products_html = build_products_carousel_html(
        tool_results,
        budget_max=float(budget_max) if isinstance(budget_max, (int, float)) else None,
        currency=currency,
        user_message=user_message,
        session_product_focus=merged.get("session_product_focus")
        if isinstance(merged.get("session_product_focus"), str)
        else None,
        last_search_products=products,
        visible_products=products,
        allow_stale_fallback=False,
    )
    message = (
        _TIMEOUT_BUDGET_PARTIAL_MESSAGE
        if budget_applied or is_budget_refinement_message(user_message)
        else _TIMEOUT_PARTIAL_MESSAGE
    )
    # Situational/empathy turns: prepend a brief apology so timeout partials still feel human.
    intent_metadata = merged.get("intent_metadata") or {}
    situational = bool(
        (isinstance(intent_metadata, dict) and intent_metadata.get("is_situational"))
        or merged.get("session_situational")
    )
    if situational:
        head = message.strip().lower()[:120]
        if not any(
            phrase in head for phrase in ("sorry", "hear that", "heartbroken", "going through")
        ):
            message = f"I'm sorry to hear you're going through this. {message}"
    if not products_html:
        return message, None

    slot_id = f"carousel-slot-timeout-{secrets.token_hex(4)}"
    response_html = render_assistant_html(message, carousel_slot_id=slot_id)
    carousel_oob = render_carousel_oob_html(products_html, carousel_slot_id=slot_id)
    return response_html, carousel_oob


def _skip_early_search_status(state: AgentState) -> bool:
    """Skip generic search status when the turn routes straight to a reply."""
    intent = state.get("intent")
    if intent in ("tracking", "checkout"):
        return True
    if state.get("specificity_band") == "clarify":
        return True
    intent_metadata = state.get("intent_metadata") or {}
    if isinstance(intent_metadata, dict) and intent_metadata.get("is_situational"):
        # Situational turns that will still search (e.g. apology flowers) need status.
        user_message = _extract_latest_user_message(state.get("messages") or [])
        if not re.search(
            r"\b(?:flower|flowers|rose|roses|bouquet|bouquets)\b",
            user_message,
            re.I,
        ):
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
    # Seed with session fields so timeout partials can budget-filter prior carousels.
    partial_state: dict[str, Any] = {
        key: state[key]
        for key in (
            "messages",
            "session_budget_max",
            "session_product_focus",
            "session_search_query",
            "session_recipient_hint",
            "session_situational",
            "last_search_products",
            "last_visible_products",
            "currency",
            "hybrid_context",
            "intent_metadata",
        )
        if key in state
    }
    turn_timeout = chat_turn_timeout_seconds()

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
        with turn_deadline(turn_timeout):
            async with asyncio.timeout(turn_timeout):
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
                        elif isinstance(payload, dict) and payload.get("type") == "carousel":
                            carousel_html = payload.get("html")
                            if isinstance(carousel_html, str) and carousel_html.strip():
                                # Ensure a provisional slot exists so OOB outerHTML can land.
                                slot_id = str(payload.get("slot_id") or "carousel-slot-provisional")
                                seed = (
                                    f'<div id="{slot_id}" class="assistant-products mt-4" '
                                    f'data-slot="product-carousel" role="region" '
                                    f'aria-label="Suggested products"></div>'
                                )
                                yield format_sse_event(seed)
                                yield format_sse_event(carousel_html, event="carousel")
                        continue

                    if mode != "updates" or not isinstance(payload, dict):
                        continue

                    for node_name, node_update in payload.items():
                        if not isinstance(node_update, dict):
                            continue
                        partial_state.update(node_update)
                        trace_node_update(node_name, node_update)

                        if node_name == "analyze_intent" and node_update.get("specificity_band") == "clarify":
                            clarify_html = _render_streaming_assistant(THINKING, pending_id, oob=True)
                            yield format_sse_event(clarify_html, event="status")
                            stream_started = True

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
                        carousel_html = node_update.get("carousel_html")
                        if isinstance(carousel_html, str) and carousel_html.strip():
                            yield format_sse_event(carousel_html, event="carousel")
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
            turn_timeout,
            thread_id or "(unknown)",
        )
        partial_html, carousel_oob = _partial_timeout_payload(
            partial_state,
            initial_state=state,
        )
        cleanup = (
            f'<div id="{pending_id}" hx-swap-oob="delete"></div>' if stream_started else ""
        )
        if carousel_oob:
            yield format_sse_event(cleanup + partial_html)
            yield format_sse_event(carousel_oob, event="carousel")
        else:
            timeout_html = _render_streaming_assistant(
                partial_html,
                pending_id,
                oob=stream_started,
            )
            yield format_sse_event(cleanup + timeout_html if cleanup else timeout_html)
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
        elif is_rate_limited(exc):
            # NIM 429 that escaped node-level handling: show friendly retry copy
            # instead of the generic hard-error banner.
            error_html = _render_streaming_assistant(
                _RATE_LIMIT_MESSAGE,
                pending_id,
                oob=stream_started,
            )
            if stream_started:
                error_html = f'<div id="{pending_id}" hx-swap-oob="delete"></div>{error_html}'
        elif is_transient_nim_error(exc):
            # Escaped APITimeoutError / connection errors: soft timeout UX + partials.
            partial_html, carousel_oob = _partial_timeout_payload(
                partial_state,
                initial_state=state,
            )
            cleanup = (
                f'<div id="{pending_id}" hx-swap-oob="delete"></div>' if stream_started else ""
            )
            if carousel_oob:
                yield format_sse_event(cleanup + partial_html)
                yield format_sse_event(carousel_oob, event="carousel")
                yield format_sse_event("", event="done")
                done_emitted = True
                return
            error_html = _render_streaming_assistant(
                partial_html or _TIMEOUT_MESSAGE,
                pending_id,
                oob=stream_started,
            )
            if stream_started:
                error_html = f"{cleanup}{error_html}"
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
            fallback_html = _render_streaming_assistant(
                _TIMEOUT_MESSAGE,
                pending_id,
                oob=False,
            )
            if cleanup:
                yield format_sse_event(cleanup)
            yield format_sse_event(fallback_html)
            yield format_sse_event("", event="done")
