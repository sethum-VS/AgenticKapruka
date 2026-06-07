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
from graphs.nodes.retrieve_hybrid_context import (
    retrieve_hybrid_context,
    route_after_analyze_intent,
)
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient


@dataclass(frozen=True, slots=True)
class ShoppingGraphDeps:
    """Injectable dependencies for shopping graph nodes (tests and chat route)."""

    kapruka_service: KaprukaService | None = None
    client_ip: str | None = None
    genai_client: genai.Client | None = None


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

    async def _analyze_intent(state: AgentState) -> dict[str, Any]:
        return await analyze_intent(state, genai_client=genai_client)

    async def _call_mcp_tools(state: AgentState) -> dict[str, Any]:
        return await call_mcp_tools(
            state,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    async def _generate_response(state: AgentState) -> dict[str, Any]:
        return await generate_response(state, genai_client=genai_client)

    graph = StateGraph(AgentState)
    graph.add_node("analyze_intent", _analyze_intent)
    graph.add_node("retrieve_hybrid_context", retrieve_hybrid_context)
    graph.add_node("call_mcp_tools", _call_mcp_tools)
    graph.add_node("generate_response", _generate_response)

    graph.add_edge(START, "analyze_intent")
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
    graph.add_edge("generate_response", END)

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
) -> AgentState:
    """Build initial AgentState for a new chat turn."""
    return cast(
        AgentState,
        {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "zep_thread_id": thread_id,
        },
    )


def append_message_state(message: str) -> AgentState:
    """Delta state for a follow-up turn; checkpoint carries prior fields."""
    return cast(AgentState, {"messages": [HumanMessage(content=message)]})
