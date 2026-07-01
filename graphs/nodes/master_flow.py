"""LangGraph node: flow-state supervisor after analyze_intent."""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.master_flow import (
    apply_master_flow_alignment,
    infer_active_flow,
    invoke_master_flow_llm,
    should_invoke_master_flow,
)
from lib.debug.trace import trace_master_flow
from lib.redis.checkout import clear_checkout_session
from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)


async def master_flow(
    state: AgentState,
    *,
    genai_client: object | None = None,
    redis_client: RedisClient | None = None,
) -> dict[str, Any]:
    """Invoke Flash flow supervisor on conflict triggers; otherwise no-op."""
    active_flow = infer_active_flow(state)
    invoke, trigger_reason = should_invoke_master_flow(state)

    if not invoke:
        logger.debug("master_flow: skipped (%s)", trigger_reason or "no_trigger")
        trace_master_flow(
            skipped=True,
            skip_reason=trigger_reason or "no_trigger",
            active_flow=active_flow,
        )
        return {
            "active_flow": active_flow,
            "master_flow_invoked": False,
            "master_flow_skip_reason": trigger_reason,
        }

    user_message = _extract_latest_user_message(state.get("messages") or [])
    alignment = await invoke_master_flow_llm(
        state,
        active_flow=active_flow,
        trigger_reason=trigger_reason or "unknown",
        genai_client=genai_client,
    )

    if alignment is None:
        trace_master_flow(
            skipped=False,
            trigger_reason=trigger_reason,
            active_flow=active_flow,
            decision="proceed",
            confidence=0.0,
            mismatch_reason="llm_failure_fail_open",
            patches_applied=False,
        )
        return {
            "active_flow": active_flow,
            "master_flow_invoked": True,
            "master_flow_decision": "proceed",
            "master_flow_mismatch_reason": "llm_failure_fail_open",
        }

    updates = apply_master_flow_alignment(
        state,
        alignment,
        user_message=user_message,
    )
    if updates.get("checkout_state") is None and state.get("checkout_state"):
        session_id = state.get("session_id") or ""
        if redis_client is not None and session_id:
            await clear_checkout_session(redis_client, session_id)
    cfg = get_settings()
    patches_applied = alignment.confidence >= cfg.master_flow_confidence_threshold and (
        bool(updates.get("master_clarifying_question"))
        or alignment.context_reset
        or alignment.resolved_intent is not None
        or bool(alignment.resolved_session_fields)
        or alignment.checkout_action in ("pause", "exit")
    )
    trace_master_flow(
        skipped=False,
        trigger_reason=trigger_reason,
        active_flow=active_flow,
        decision=alignment.decision,
        confidence=alignment.confidence,
        mismatch_reason=alignment.mismatch_reason,
        patches_applied=patches_applied,
    )
    logger.info(
        "master_flow: decision=%s confidence=%.2f trigger=%s",
        alignment.decision,
        alignment.confidence,
        trigger_reason,
    )
    return updates
