"""MCP HTTP JSON-RPC client for Kapruka tools via the official MCP Python SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://mcp.kapruka.com/mcp"
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 0.5
# Non-idempotent write tools must not be retried on transient 5xx/timeouts.
NON_RETRYABLE_TOOLS = frozenset({"kapruka_create_order"})


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient HTTP 5xx and timeout failures."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    if isinstance(exc, McpError):
        return exc.error.code == httpx.codes.REQUEST_TIMEOUT
    if isinstance(exc, ExceptionGroup):
        return any(_is_retryable(nested) for nested in exc.exceptions)
    return False


def _extract_result_text(result: CallToolResult) -> str:
    """Return concatenated text blocks from an MCP tool result."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent) or (
            getattr(block, "type", None) == "text" and hasattr(block, "text")
        ):
            parts.append(block.text)
    return "\n".join(parts)


class MCPHttpClient:
    """Async Kapruka MCP client over Streamable HTTP JSON-RPC."""

    def __init__(
        self,
        url: str = DEFAULT_MCP_URL,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._injected_httpx_client = httpx_client
        self._owns_httpx_client = False
        self._httpx_client: httpx.AsyncClient | None = None

    @classmethod
    async def connect(
        cls,
        url: str = DEFAULT_MCP_URL,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> MCPHttpClient:
        """Create a connected MCP HTTP client."""
        client = cls(url, httpx_client=httpx_client)
        await client._ensure_httpx_client()
        return client

    async def call_tool(self, name: str, params: dict[str, Any] | None = None) -> str:
        """Invoke an MCP tool and return the raw text payload."""
        arguments = {"params": params or {}}

        if name in NON_RETRYABLE_TOOLS:
            return await self._call_tool_once(name, arguments)

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                return await self._call_tool_once(name, arguments)
            except Exception as exc:
                if not _is_retryable(exc) or attempt >= MAX_RETRY_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "MCP call_tool %s attempt %s/%s failed: %s",
                    name,
                    attempt + 1,
                    MAX_RETRY_ATTEMPTS,
                    exc,
                )
                delay = RETRY_BASE_DELAY_SECONDS * (2**attempt)
                await asyncio.sleep(delay)

        raise AssertionError("unreachable")  # loop always returns or raises

    async def close(self) -> None:
        """Close the owned HTTP client; safe to call multiple times."""
        if not self._owns_httpx_client or self._httpx_client is None:
            return
        await self._httpx_client.aclose()
        self._httpx_client = None
        self._owns_httpx_client = False
        logger.debug("MCP HTTP client closed")

    async def ping(self) -> bool:
        """Lightweight connectivity check — initialize MCP session without a tool call."""
        httpx_client = await self._ensure_httpx_client()
        async with (
            streamable_http_client(self._url, http_client=httpx_client) as (
                read_stream,
                write_stream,
                _get_session_id,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
        return True

    async def _ensure_httpx_client(self) -> httpx.AsyncClient:
        if self._httpx_client is not None:
            return self._httpx_client

        if self._injected_httpx_client is not None:
            self._httpx_client = self._injected_httpx_client
            return self._httpx_client

        self._httpx_client = create_mcp_http_client()
        self._owns_httpx_client = True
        return self._httpx_client

    async def _call_tool_once(self, name: str, arguments: dict[str, Any]) -> str:
        httpx_client = await self._ensure_httpx_client()
        async with (
            streamable_http_client(self._url, http_client=httpx_client) as (
                read_stream,
                write_stream,
                _get_session_id,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return _extract_result_text(result)
