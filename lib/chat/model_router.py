"""Route specialized chat LLM calls to NVIDIA NIM model."""

from __future__ import annotations

from app.config import Settings, get_settings

# Single NVIDIA NIM model — LoRA fine-tuning not applicable on NIM free tier.
FLASH_MODEL = "z-ai/glm-5.2"


def select_specialized_model(*, settings: Settings | None = None) -> str:
    """Return the NVIDIA NIM LLM model for specialized chat tasks."""
    cfg = settings or get_settings()
    return cfg.nvidia_llm_model


def select_intent_model(*, settings: Settings | None = None) -> str:
    """Model for ``analyze_intent`` structured classification."""
    return select_specialized_model(settings=settings)


def select_rewrite_model(*, settings: Settings | None = None) -> str:
    """Model for occasion-aware discovery query rewrite."""
    return select_specialized_model(settings=settings)
