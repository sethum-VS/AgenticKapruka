"""Select NVIDIA NIM model for the shopping graph agent."""

from __future__ import annotations

from app.config import get_settings
from graphs.state import AgentState, ModelTier

# Single NVIDIA NIM model for all LLM tasks — no flash/pro tiering.
NVIDIA_MODEL: str = get_settings().nvidia_llm_model if get_settings else "z-ai/glm-5.2"

# Legacy aliases for backward compatibility with state/tests.
FLASH_MODEL = "z-ai/glm-5.2"
PRO_MODEL = "z-ai/glm-5.2"


def select_model_tier(state: AgentState) -> ModelTier:
    """Return model tier — always flash with single-model NVIDIA NIM."""
    return "flash"


def select_model(state: AgentState) -> str:
    """Return NVIDIA NIM model name for the current agent turn."""
    cfg = get_settings()
    return cfg.nvidia_llm_model
