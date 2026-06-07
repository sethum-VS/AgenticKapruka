"""Load Zep memory facts into AgentState before intent analysis."""

from __future__ import annotations

import logging
from typing import Any

from graphs.state import AgentState
from lib.zep.client import ZepClient
from lib.zep.memory import get_session_memory_facts

logger = logging.getLogger(__name__)


async def load_zep_memory(
    state: AgentState,
    *,
    zep_client: ZepClient | None = None,
) -> dict[str, Any]:
    """LangGraph node: fetch last Zep facts for the session thread."""
    thread_id = state.get("zep_thread_id")
    if not thread_id or zep_client is None:
        logger.debug("load_zep_memory: skipped (no thread_id or zep_client)")
        return {"zep_memory_facts": []}

    facts = await get_session_memory_facts(zep_client, thread_id)
    logger.info("load_zep_memory: loaded %d facts for thread %s", len(facts), thread_id)
    return {"zep_memory_facts": facts}
