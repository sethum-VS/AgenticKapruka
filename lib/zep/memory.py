"""Zep memory read/write helpers for the shopping graph."""

from __future__ import annotations

import logging
from typing import Final

from zep_python.types.memory import Memory
from zep_python.types.message import Message

from lib.zep.client import ZepClient

logger = logging.getLogger(__name__)

DEFAULT_FACT_LIMIT: Final = 10


def extract_memory_facts(memory: Memory, *, limit: int = DEFAULT_FACT_LIMIT) -> list[str]:
    """Return up to ``limit`` most recent fact strings from a Zep memory payload."""
    raw_facts = memory.facts or []
    fact_strings = [str(item) for item in raw_facts if item]
    if len(fact_strings) <= limit:
        return fact_strings
    return fact_strings[-limit:]


def format_memory_facts_block(facts: list[str]) -> str:
    """Format fact strings for injection into LLM system prompts."""
    if not facts:
        return ""
    lines = "\n".join(f"- {fact}" for fact in facts)
    return f"\n\nPrior session facts (context only):\n{lines}"


async def get_session_memory_facts(
    zep_client: ZepClient,
    session_id: str,
    *,
    limit: int = DEFAULT_FACT_LIMIT,
) -> list[str]:
    """Load the last ``limit`` Zep memory facts for a session thread."""
    try:
        memory = await zep_client.get_memory(session_id)
    except Exception as exc:
        logger.warning("Failed to load Zep memory for session %s: %s", session_id, exc)
        return []
    return extract_memory_facts(memory, limit=limit)


async def append_session_messages(
    zep_client: ZepClient,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append a user/assistant turn to Zep session memory."""
    messages = [
        Message(role="user", role_type="user", content=user_message),
        Message(role="assistant", role_type="assistant", content=assistant_message),
    ]
    await zep_client.add_messages(session_id, messages)
