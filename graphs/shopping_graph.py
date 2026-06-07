"""Main shopping LangGraph — analyze → hybrid context → MCP tools → response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from google import genai
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graphs.nodes.analyze_intent import analyze_intent
from graphs.nodes.call_mcp_tools import call_mcp_tools
from graphs.nodes.generate_response import generate_response
from graphs.nodes.load_zep_memory import load_zep_memory
from graphs.nodes.retrieve_hybrid_context import (
    retrieve_hybrid_context,
    route_after_analyze_intent,
)
from graphs.nodes.zep_memory_write import zep_memory_write
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.neo4j.client import Neo4jClient
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient
from lib.zep.client import ZepClient


@dataclass(frozen=True, slots=True)
class ShoppingGraphDeps:
    """Injectable dependencies for shopping graph nodes (tests and chat route)."""

    kapruka_service: KaprukaService | None = None
    client_ip: str | None = None
    genai_client: genai.Client | None = None
    neo4j_client: Neo4jClient | None = None
    zep_client: ZepClient | None = None


def build_shopping_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    deps: ShoppingGraphDeps | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the shopping StateGraph with optional Redis checkpointer."""
    resolved = deps or ShoppingGraphDeps()
    genai_client = resolved.genai_client
    kapruka_service = resolved.kapruka_service
    client_ip = resolved.client_ip
    neo4j_client = resolved.neo4j_client
    zep_client = resolved.zep_client

    async def _load_zep_memory(state: AgentState) -> dict[str, Any]:
        return await load_zep_memory(state, zep_client=zep_client)

    async def _analyze_intent(state: AgentState) -> dict[str, Any]:
        return await analyze_intent(state, genai_client=genai_client)

    async def _retrieve_hybrid_context(state: AgentState) -> dict[str, Any]:
        return await retrieve_hybrid_context(
            state,
            zep_client=zep_client,
            neo4j_client=neo4j_client,
        )

    async def _call_mcp_tools(state: AgentState) -> dict[str, Any]:
        return await call_mcp_tools(
            state,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    async def _generate_response(state: AgentState) -> dict[str, Any]:
        return await generate_response(state, genai_client=genai_client)

    async def _zep_memory_write(state: AgentState) -> dict[str, Any]:
        return await zep_memory_write(state, zep_client=zep_client)

    graph = StateGraph(AgentState)
    graph.add_node("load_zep_memory", _load_zep_memory)
    graph.add_node("analyze_intent", _analyze_intent)
    graph.add_node("retrieve_hybrid_context", _retrieve_hybrid_context)
    graph.add_node("call_mcp_tools", _call_mcp_tools)
    graph.add_node("generate_response", _generate_response)
    graph.add_node("zep_memory_write", _zep_memory_write)

    graph.add_edge(START, "load_zep_memory")
    graph.add_edge("load_zep_memory", "analyze_intent")
    graph.add_conditional_edges(
        "analyze_intent",
        route_after_analyze_intent,
        {
            "retrieve_hybrid_context": "retrieve_hybrid_context",
            "call_mcp_tools": "call_mcp_tools",
        },
    )
    graph.add_edge("retrieve_hybrid_context", "call_mcp_tools")
    graph.add_edge("call_mcp_tools", "generate_response")
    graph.add_edge("generate_response", "zep_memory_write")
    graph.add_edge("zep_memory_write", END)

    return graph.compile(checkpointer=checkpointer)


async def get_shopping_graph(
    redis_client: RedisClient,
    *,
    deps: ShoppingGraphDeps | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Factory: Redis checkpointer + compiled shopping graph."""
    checkpointer = await get_checkpointer(redis_client)
    return build_shopping_graph(checkpointer=checkpointer, deps=deps)


def initial_shopping_state(
    *,
    message: str,
    session_id: str,
    thread_id: str | None = None,
    zep_thread_id: str | None = None,
    currency: str | None = None,
) -> AgentState:
    """Build initial AgentState for a new chat turn."""
    resolved_thread = zep_thread_id if zep_thread_id is not None else thread_id
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "zep_thread_id": resolved_thread,
    }
    if currency is not None:
        state["currency"] = currency
    return cast(AgentState, state)


def append_message_state(message: str, *, currency: str | None = None) -> AgentState:
    """Delta state for a follow-up turn; checkpoint carries prior fields."""
    delta: dict[str, Any] = {"messages": [HumanMessage(content=message)]}
    if currency is not None:
        delta["currency"] = currency
    return cast(AgentState, delta)
