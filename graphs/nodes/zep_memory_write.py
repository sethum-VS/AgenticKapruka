"""Persist user/assistant turn to Zep after response generation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.zep.client import ZepClient
from lib.zep.memory import append_session_messages

logger = logging.getLogger(__name__)

_zep_write_tasks: set[asyncio.Task[None]] = set()


async def _persist_turn(
    zep_client: ZepClient,
    thread_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    try:
        await append_session_messages(
            zep_client,
            thread_id,
            user_message,
            assistant_message,
        )
    except Exception as exc:
        logger.warning(
            "zep_memory_write: failed to persist turn for thread %s: %s",
            thread_id,
            exc,
        )
        return
    logger.info("zep_memory_write: persisted turn for thread %s", thread_id)


async def zep_memory_write(
    state: AgentState,
    *,
    zep_client: ZepClient | None = None,
) -> dict[str, Any]:
    """LangGraph node: append the completed turn to Zep session memory."""
    thread_id = state.get("zep_thread_id")
    if not thread_id or zep_client is None:
        logger.debug("zep_memory_write: skipped (no thread_id or zep_client)")
        return {}

    user_message = _extract_latest_user_message(state.get("messages") or [])
    assistant_message = (state.get("assistant_message") or "").strip()
    if not user_message.strip() or not assistant_message:
        logger.debug("zep_memory_write: skipped (empty user or assistant message)")
        return {}

    task = asyncio.create_task(
        _persist_turn(zep_client, thread_id, user_message, assistant_message),
    )
    _zep_write_tasks.add(task)
    task.add_done_callback(_zep_write_tasks.discard)
    return {}
