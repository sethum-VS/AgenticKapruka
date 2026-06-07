"""Dependency builders for the chat streaming route."""

from __future__ import annotations

import logging

from google import genai
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from starlette.requests import Request

from app.config import get_settings
from graphs.shopping_graph import ShoppingGraphDeps, get_shopping_graph
from graphs.state import AgentState
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.service import KaprukaService
from lib.redis.client import RedisClient
from lib.zep.client import ZepClient

logger = logging.getLogger(__name__)


def client_ip_from_request(request: Request) -> str:
    """Best-effort client IP for Kapruka rate limiting."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", maxsplit=1)[0].strip()
    if request.client is not None:
        return request.client.host
    return "127.0.0.1"


async def ensure_mcp_client(request: Request) -> MCPHttpClient:
    """Lazy-connect Kapruka MCP client on application state."""
    existing: MCPHttpClient | None = getattr(request.app.state, "mcp_client", None)
    if existing is not None:
        return existing
    settings = get_settings()
    client = await MCPHttpClient.connect(settings.kapruka_mcp_url)
    request.app.state.mcp_client = client
    logger.info("Kapruka MCP client connected")
    return client


async def ensure_kapruka_service(request: Request, redis_client: RedisClient) -> KaprukaService:
    """Lazy-build KaprukaService facade on application state."""
    existing: KaprukaService | None = getattr(request.app.state, "kapruka_service", None)
    if existing is not None:
        return existing
    mcp_client = await ensure_mcp_client(request)
    service = KaprukaService(redis_client, mcp_client)
    request.app.state.kapruka_service = service
    return service


def zep_client_from_app(request: Request) -> ZepClient | None:
    """Return Zep client when wired during lifespan."""
    return getattr(request.app.state, "zep", None)


async def build_shopping_graph_deps(
    request: Request,
    redis_client: RedisClient,
) -> ShoppingGraphDeps:
    """Assemble injectable graph dependencies for a chat turn."""
    kapruka_service = await ensure_kapruka_service(request, redis_client)
    settings = get_settings()
    genai_client = genai.Client(api_key=settings.google_api_key)
    return ShoppingGraphDeps(
        kapruka_service=kapruka_service,
        client_ip=client_ip_from_request(request),
        genai_client=genai_client,
        zep_client=zep_client_from_app(request),
    )


async def get_compiled_chat_graph(
    redis_client: RedisClient,
    *,
    deps: ShoppingGraphDeps,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile shopping graph with Redis checkpointer (patchable in tests)."""
    return await get_shopping_graph(redis_client, deps=deps)


async def resolve_turn_state(
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    *,
    message: str,
    session_id: str,
    zep_thread_id: str | None,
    config: RunnableConfig,
) -> AgentState:
    """Use checkpoint thread state for follow-ups; seed session on first turn."""
    from graphs.shopping_graph import append_message_state, initial_shopping_state

    snapshot = await graph.aget_state(config)
    if snapshot.values:
        return append_message_state(message)
    return initial_shopping_state(
        message=message,
        session_id=session_id,
        zep_thread_id=zep_thread_id,
    )
