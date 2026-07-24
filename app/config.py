"""Application configuration from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    redis_url: str = Field(..., description="Memorystore or local Redis URL")
    neo4j_uri: str = Field(..., description="Neo4j AuraDB bolt URI")
    neo4j_user: str = Field(..., min_length=1)
    neo4j_password: str = Field(..., min_length=1)
    zep_api_key: str = Field(..., min_length=1)

    # ── NVIDIA NIM ────────────────────────────────────────────────────────
    nvidia_api_key: str = Field(..., min_length=1, description="NVIDIA NIM API key")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM OpenAI-compatible base URL",
    )
    nvidia_llm_model: str = Field(
        default="z-ai/glm-5.2",
        description="NVIDIA NIM LLM model for all text generation tasks",
    )
    nvidia_embedding_model: str = Field(
        default="nvidia/nv-embed-v1",
        description="NVIDIA NIM embedding model for GraphRAG vectors",
    )
    nvidia_rate_limit_rpm: int = Field(
        default=30,
        ge=1,
        description="Max requests per minute for NVIDIA NIM free-tier safety",
    )
    chat_turn_timeout_seconds: int = Field(
        default=120,
        ge=30,
        description="Wall-clock timeout for a single chat SSE turn before graceful fallback",
    )
    nvidia_vector_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for nv-embed-v1 (4096d) vector matches. "
            "Baseline 0.75 — tune via evals/ragas_eval.py after re-indexing."
        ),
    )

    kapruka_mcp_url: str = Field(
        default="https://mcp.kapruka.com/mcp",
        description="Kapruka MCP JSON-RPC endpoint",
    )
    session_secret: str = Field(..., min_length=32)
    reranker_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum cross-encoder score to keep Occasion/Category traversal nodes",
    )
    master_flow_enabled: bool = Field(
        default=True,
        description="Enable flow-state supervisor after analyze_intent on conflict triggers",
    )
    master_flow_long_session_turns: int = Field(
        default=8,
        ge=2,
        description="Human turn count before long-session drift can trigger master_flow",
    )
    master_flow_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum Flash confidence before master_flow patches are applied",
    )

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        if not value.startswith(("redis://", "rediss://")):
            msg = "REDIS_URL must use redis:// or rediss:// scheme"
            raise ValueError(msg)
        return value

    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, value: str) -> str:
        if not value.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
            msg = "NEO4J_URI must be a bolt or neo4j URI"
            raise ValueError(msg)
        return value

    @field_validator("kapruka_mcp_url")
    @classmethod
    def validate_kapruka_mcp_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            msg = "KAPRUKA_MCP_URL must be an HTTP(S) URL"
            raise ValueError(msg)
        return value

    @field_validator(
        "neo4j_user",
        "neo4j_password",
        "zep_api_key",
        "nvidia_api_key",
        "session_secret",
        mode="after",
    )
    @classmethod
    def validate_required_production_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "Production configuration value must not be empty"
            raise ValueError(msg)
        return stripped

    @field_validator("nvidia_base_url")
    @classmethod
    def validate_nvidia_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            msg = "NVIDIA_BASE_URL must be an HTTP(S) URL"
            raise ValueError(msg)
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
