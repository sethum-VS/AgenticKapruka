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
    google_api_key: str = Field(..., min_length=1)
    gcp_project_id: str = Field(..., min_length=1)
    gcp_location: str = Field(..., min_length=1)
    kapruka_mcp_url: str = Field(
        default="https://mcp.kapruka.com/mcp",
        description="Kapruka MCP JSON-RPC endpoint",
    )
    session_secret: str = Field(..., min_length=32)

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
        "google_api_key",
        "gcp_project_id",
        "gcp_location",
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


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
