"""Async Zep Cloud client wrapper with session management and health checks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import httpx
from zep_python.client import AsyncZep
from zep_python.types import Session, SessionListResponse
from zep_python.types.memory import Memory
from zep_python.types.message import Message
from zep_python.types.session_search_response import SessionSearchResponse
from zep_python.types.success_response import SuccessResponse

SearchScope = Literal["messages", "summary", "facts"]

logger = logging.getLogger(__name__)

_DEFAULT_ZEP_BASE_URL = "https://api.getzep.com"
_DEFAULT_TIMEOUT = 30.0


class ZepClient:
    """Wrap zep-python AsyncZep with session helpers and API key health checks."""

    def __init__(
        self,
        api_key: str,
        *,
        client: AsyncZep | None = None,
        base_url: str = _DEFAULT_ZEP_BASE_URL,
        request_timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._client = client
        self._injected_httpx_client: httpx.AsyncClient | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._owns_httpx_client = False

    @classmethod
    async def connect(
        cls,
        api_key: str,
        *,
        base_url: str = _DEFAULT_ZEP_BASE_URL,
        request_timeout: float = _DEFAULT_TIMEOUT,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> ZepClient:
        """Create a connected client for Zep Cloud."""
        instance = cls(api_key, base_url=base_url, request_timeout=request_timeout)
        if httpx_client is not None:
            instance._injected_httpx_client = httpx_client
            instance._httpx_client = httpx_client
        else:
            instance._httpx_client = httpx.AsyncClient(timeout=request_timeout)
            instance._owns_httpx_client = True
        instance._client = AsyncZep(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            httpx_client=instance._httpx_client,
        )
        return instance

    @property
    def sdk(self) -> AsyncZep:
        """Underlying zep-python AsyncZep client for memory and user modules."""
        if self._client is None:
            msg = "ZepClient is not connected; call connect() first"
            raise RuntimeError(msg)
        return self._client

    async def list_sessions(
        self,
        *,
        page_number: int = 1,
        page_size: int = 10,
        order_by: str | None = None,
        asc: bool | None = None,
    ) -> SessionListResponse:
        """List memory sessions with optional pagination."""
        return await self.sdk.memory.list_sessions(
            page_number=page_number,
            page_size=page_size,
            order_by=order_by,
            asc=asc,
        )

    async def create_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new Zep memory session."""
        if user_id is None and metadata is None:
            return await self.sdk.memory.add_session(session_id=session_id)
        if metadata is None:
            return await self.sdk.memory.add_session(session_id=session_id, user_id=user_id)
        if user_id is None:
            return await self.sdk.memory.add_session(session_id=session_id, metadata=metadata)
        return await self.sdk.memory.add_session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )

    async def get_memory(
        self,
        session_id: str,
        *,
        lastn: int | None = None,
    ) -> Memory:
        """Return memory (summary, messages, facts) for a session thread."""
        return await self.sdk.memory.get(session_id, lastn=lastn)

    async def search_session_memory(
        self,
        *,
        session_ids: Sequence[str] | None = None,
        text: str | None = None,
        search_scope: SearchScope = "facts",
        limit: int | None = 10,
    ) -> SessionSearchResponse:
        """Semantic search across Zep session memory (facts, messages, or summary)."""
        return await self.sdk.memory.search_sessions(
            session_ids=session_ids,
            text=text,
            search_scope=search_scope,
            limit=limit,
        )

    async def add_messages(
        self,
        session_id: str,
        messages: Sequence[Message],
    ) -> SuccessResponse:
        """Append chat messages to a session's Zep memory."""
        return await self.sdk.memory.add(session_id, messages=messages)

    async def health_check(self) -> bool:
        """Return True when Zep Cloud accepts the configured API key."""
        try:
            await self.sdk.memory.list_sessions(page_size=1, page_number=1)
        except Exception as exc:
            logger.warning("Zep health check failed: %s", exc)
            return False
        return True

    async def close(self) -> None:
        """Close the owned httpx client; safe to call multiple times."""
        if self._client is None:
            return
        if self._owns_httpx_client and self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
            self._owns_httpx_client = False
        self._client = None
        logger.debug("Zep client closed")
