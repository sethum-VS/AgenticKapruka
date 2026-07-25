"""Factory for NVIDIA NIM OpenAI-compatible clients (primary + backup)."""

from __future__ import annotations

from typing import Literal

from openai import OpenAI

from app.config import Settings, get_settings

NimKeyRole = Literal["primary", "backup"]

_clients: dict[NimKeyRole, OpenAI] = {}


def _api_key_for_role(cfg: Settings, role: NimKeyRole) -> str | None:
    if role == "primary":
        return cfg.nvidia_api_key
    backup = (cfg.nvidia_api_key_backup or "").strip()
    return backup or None


def create_nvidia_client(
    *,
    settings: Settings | None = None,
    role: NimKeyRole = "primary",
) -> OpenAI:
    """Return a cached NVIDIA NIM OpenAI client for the given key role.

    Points at ``NVIDIA_BASE_URL`` (default https://integrate.api.nvidia.com/v1)
    with ``NVIDIA_API_KEY`` (primary) or ``NVIDIA_API_KEY_BACKUP`` (backup).
    """
    if role in _clients:
        return _clients[role]
    cfg = settings or get_settings()
    api_key = _api_key_for_role(cfg, role)
    if not api_key:
        msg = f"NVIDIA NIM {role} API key is not configured"
        raise ValueError(msg)
    client = OpenAI(
        base_url=cfg.nvidia_base_url,
        api_key=api_key,
        timeout=30.0,
    )
    _clients[role] = client
    return client


def has_backup_nvidia_client(*, settings: Settings | None = None) -> bool:
    """True when ``NVIDIA_API_KEY_BACKUP`` is configured."""
    cfg = settings or get_settings()
    return bool((cfg.nvidia_api_key_backup or "").strip())


def reset_client() -> None:
    """Drop cached clients (for tests)."""
    _clients.clear()
