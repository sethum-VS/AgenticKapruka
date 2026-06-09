"""Async Zep Cloud v3 client wrapper with thread management and health checks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx
from zep_cloud.client import AsyncZep
from zep_cloud.types.graph_search_results import GraphSearchResults
from zep_cloud.types.message import Message
from zep_cloud.types.thread import Thread
from zep_cloud.types.thread_context_response import ThreadContextResponse
from zep_cloud.types.thread_list_response import ThreadListResponse

logger = logging.getLogger(__name__)

_DEFAULT_ZEP_BASE_URL = "https://api.getzep.com/api/v2"
_DEFAULT_TIMEOUT = 30.0
_USER_EXISTS_MARKER = "user already exists"


class ZepClient:
    """Wrap zep-cloud AsyncZep with thread helpers and API key health checks."""

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
        """Underlying zep-cloud AsyncZep client."""
        if self._client is None:
            msg = "ZepClient is not connected; call connect() first"
            raise RuntimeError(msg)
        return self._client

    async def list_threads(
        self,
        *,
        page_number: int = 1,
        page_size: int = 10,
        order_by: str | None = None,
        asc: bool | None = None,
    ) -> ThreadListResponse:
        """List Zep threads with optional pagination."""
        return await self.sdk.thread.list_all(
            page_number=page_number,
            page_size=page_size,
            order_by=order_by,
            asc=asc,
        )

    async def ensure_user(self, user_id: str) -> None:
        """Create a Zep user when missing; ignore duplicate-user errors."""
        try:
            await self.sdk.user.add(user_id=user_id)
        except Exception as exc:
            message = str(exc).lower()
            if _USER_EXISTS_MARKER not in message:
                raise

    async def create_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Thread:
        """Create a Zep thread for a browser session (session_id maps to thread_id)."""
        del metadata  # v3 thread.create does not accept metadata
        resolved_user_id = user_id or session_id
        await self.ensure_user(resolved_user_id)
        return await self.sdk.thread.create(
            thread_id=session_id,
            user_id=resolved_user_id,
        )

    async def get_user_context(self, session_id: str) -> ThreadContextResponse:
        """Return the Zep context block for a thread."""
        return await self.sdk.thread.get_user_context(thread_id=session_id)

    async def search_graph(
        self,
        *,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> GraphSearchResults:
        """Search the user knowledge graph for relevant facts."""
        return await self.sdk.graph.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )

    async def add_messages(
        self,
        session_id: str,
        messages: Sequence[Message],
    ) -> Any:
        """Append chat messages to a Zep thread."""
        return await self.sdk.thread.add_messages(
            thread_id=session_id,
            messages=list(messages),
        )

    async def health_check(self) -> bool:
        """Return True when Zep Cloud accepts the configured API key."""
        try:
            await self.sdk.thread.list_all(page_size=1, page_number=1)
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
