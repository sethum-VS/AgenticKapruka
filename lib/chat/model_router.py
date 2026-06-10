"""Route specialized chat LLM calls to base Flash or a LoRA-tuned Vertex endpoint."""

from __future__ import annotations

from app.config import Settings, get_settings

FLASH_MODEL = "gemini-2.5-flash"


def build_lora_endpoint_resource(
    project_id: str,
    location: str,
    endpoint_id: str,
) -> str:
    """Build the Vertex AI endpoint resource name for google-genai ``model``."""
    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id.strip()}"


def select_specialized_model(*, settings: Settings | None = None) -> str:
    """Return LoRA Vertex endpoint when configured, otherwise base Flash."""
    cfg = settings or get_settings()
    endpoint_id = cfg.kapruka_lora_endpoint_id
    if endpoint_id:
        return build_lora_endpoint_resource(
            cfg.gcp_project_id,
            cfg.gcp_location,
            endpoint_id,
        )
    return FLASH_MODEL


def select_intent_model(*, settings: Settings | None = None) -> str:
    """Model for ``analyze_intent`` structured classification."""
    return select_specialized_model(settings=settings)


def select_rewrite_model(*, settings: Settings | None = None) -> str:
    """Model for occasion-aware discovery query rewrite."""
    return select_specialized_model(settings=settings)
