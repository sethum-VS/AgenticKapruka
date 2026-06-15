"""Zep memory read/write helpers for the shopping graph."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

from zep_cloud.types.graph_search_results import GraphSearchResults
from zep_cloud.types.message import Message

from lib.zep.client import ZepClient

logger = logging.getLogger(__name__)

DEFAULT_FACT_LIMIT: Final = 10
_BULLET_PREFIX_RE = re.compile(r"^[-*•]\s+")
_RECIPIENT_ENTITY_RE = re.compile(
    r"\b(?:mom|mother|mum|mama|amma|dad|father|papa|thatha|wife|husband|"
    r"girlfriend|boyfriend|partner|fianc[eé]e|sister|brother|son|daughter|"
    r"grandma|grandmother|grandpa|grandfather|aunt|uncle|nana|nanna)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ZepMemory:
    """Normalized Zep memory payload for graph nodes and prompts."""

    facts: list[str]
    summary: str = ""
    context: str | None = None


def facts_from_context(context: str | None) -> list[str]:
    """Split a Zep context block into discrete fact strings."""
    if context is None or not context.strip():
        return []

    lines: list[str] = []
    for raw_line in context.splitlines():
        stripped = _BULLET_PREFIX_RE.sub("", raw_line.strip())
        if stripped:
            lines.append(stripped)

    if lines:
        return lines
    return [context.strip()]


def facts_from_graph_search(results: GraphSearchResults) -> list[str]:
    """Collect fact strings from a Zep graph search response."""
    facts = facts_from_context(results.context)
    for edge in results.edges or []:
        if edge.fact and edge.fact not in facts:
            facts.append(edge.fact)
    return facts


def extract_memory_facts(memory: ZepMemory, *, limit: int = DEFAULT_FACT_LIMIT) -> list[str]:
    """Return up to ``limit`` most recent fact strings from a Zep memory payload."""
    fact_strings = [item for item in memory.facts if item]
    if len(fact_strings) <= limit:
        return fact_strings
    return fact_strings[-limit:]


def message_references_recipient(text: str) -> bool:
    """Return True when the user message names a gift recipient or relation."""
    return bool(_RECIPIENT_ENTITY_RE.search(text))


def scope_memory_facts_for_turn(
    facts: list[str],
    user_message: str,
) -> list[str]:
    """Drop recipient-specific Zep facts when the current turn does not mention them."""
    if not facts or message_references_recipient(user_message):
        return facts
    return [fact for fact in facts if not _RECIPIENT_ENTITY_RE.search(fact)]


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
    """Load Zep memory facts for a session thread."""
    try:
        context_response = await zep_client.get_user_context(session_id)
    except Exception as exc:
        logger.warning("Failed to load Zep memory for session %s: %s", session_id, exc)
        return []

    memory = ZepMemory(
        facts=facts_from_context(context_response.context),
        context=context_response.context,
    )
    return extract_memory_facts(memory, limit=limit)


async def append_session_messages(
    zep_client: ZepClient,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append a user/assistant turn to Zep thread memory."""
    messages = [
        Message(role="user", content=user_message),
        Message(role="assistant", content=assistant_message),
    ]
    await zep_client.add_messages(session_id, messages)
