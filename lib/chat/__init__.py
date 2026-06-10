"""Chat streaming helpers for LangGraph SSE responses."""

from __future__ import annotations

__all__ = ["format_sse_event", "iter_chat_sse_events"]


def __getattr__(name: str) -> object:
    if name == "format_sse_event":
        from lib.chat.sse import format_sse_event

        return format_sse_event
    if name == "iter_chat_sse_events":
        from lib.chat.streaming import iter_chat_sse_events

        return iter_chat_sse_events
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
