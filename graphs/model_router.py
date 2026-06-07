"""Select Gemini model tier based on checkout context and tool-call depth."""

from __future__ import annotations

from graphs.state import AgentState, ModelTier

FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-pro"

_TOOL_CALL_PRO_THRESHOLD = 3


def select_model_tier(state: AgentState) -> ModelTier:
    """Return flash or pro tier from explicit override and escalation rules."""
    tier: ModelTier | None = state.get("model_tier")
    if tier == "pro":
        return "pro"

    checkout_state = state.get("checkout_state")
    if checkout_state == "review":
        return "pro"

    tool_call_count = state.get("tool_call_count") or 0
    if tool_call_count > _TOOL_CALL_PRO_THRESHOLD:
        return "pro"

    return "flash"


def select_model(state: AgentState) -> str:
    """Return Gemini model name for the current agent turn."""
    tier = select_model_tier(state)
    if tier == "pro":
        return PRO_MODEL
    return FLASH_MODEL
