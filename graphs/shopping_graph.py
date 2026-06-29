"""Main shopping LangGraph — analyze → hybrid context → agent loop → response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from google import genai
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graphs.nodes.agent_loop import agent_loop
from graphs.nodes.analyze_intent import analyze_intent
from graphs.nodes.call_mcp_tools import call_mcp_tools
from graphs.nodes.execute_cart_action import execute_cart_action
from graphs.nodes.generate_response import generate_response
from graphs.nodes.load_zep_memory import load_zep_memory
from graphs.nodes.master_flow import master_flow
from graphs.nodes.resolve_cart_product import resolve_cart_product
from graphs.nodes.resolve_delivery_context import (
    resolve_delivery_context,
    route_after_resolve_delivery_context,
)
from graphs.nodes.retrieve_hybrid_context import retrieve_hybrid_context
from graphs.nodes.run_checkout_graph import run_checkout_graph
from graphs.nodes.zep_memory_write import zep_memory_write
from graphs.state import AgentState
from lib.chat.routing import route_after_master_flow
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
    redis_client: RedisClient | None = None


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
    redis_client = resolved.redis_client

    async def _load_zep_memory(state: AgentState) -> dict[str, Any]:
        return await load_zep_memory(state, zep_client=zep_client)

    async def _analyze_intent(state: AgentState) -> dict[str, Any]:
        return await analyze_intent(
            state,
            genai_client=genai_client,
            redis_client=redis_client,
        )

    async def _retrieve_hybrid_context(state: AgentState) -> dict[str, Any]:
        return await retrieve_hybrid_context(
            state,
            zep_client=zep_client,
            neo4j_client=neo4j_client,
            redis_client=redis_client,
        )

    async def _call_mcp_tools(state: AgentState) -> dict[str, Any]:
        return await call_mcp_tools(
            state,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    async def _agent_loop(state: AgentState) -> dict[str, Any]:
        return await agent_loop(
            state,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
            genai_client=genai_client,
        )

    async def _generate_response(state: AgentState) -> dict[str, Any]:
        return await generate_response(state, genai_client=genai_client)

    async def _zep_memory_write(state: AgentState) -> dict[str, Any]:
        return await zep_memory_write(state, zep_client=zep_client)

    async def _run_checkout_graph(state: AgentState) -> dict[str, Any]:
        return await run_checkout_graph(
            state,
            redis_client=redis_client,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    async def _master_flow(state: AgentState) -> dict[str, Any]:
        updates = await master_flow(
            state,
            genai_client=genai_client,
            redis_client=redis_client,
        )
        return updates

    async def _resolve_cart_product(state: AgentState) -> dict[str, Any]:
        return await resolve_cart_product(
            state,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    async def _execute_cart_action(state: AgentState) -> dict[str, Any]:
        return await execute_cart_action(
            state,
            redis_client=redis_client,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    async def _resolve_delivery_context(state: AgentState) -> dict[str, Any]:
        return await resolve_delivery_context(
            state,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
            genai_client=genai_client,
        )

    graph = StateGraph(AgentState)
    graph.add_node("load_zep_memory", _load_zep_memory)
    graph.add_node("analyze_intent", _analyze_intent)
    graph.add_node("master_flow", _master_flow)
    graph.add_node("retrieve_hybrid_context", _retrieve_hybrid_context)
    graph.add_node("call_mcp_tools", _call_mcp_tools)
    graph.add_node("agent_loop", _agent_loop)
    graph.add_node("generate_response", _generate_response)
    graph.add_node("run_checkout_graph", _run_checkout_graph)
    graph.add_node("resolve_cart_product", _resolve_cart_product)
    graph.add_node("resolve_delivery_context", _resolve_delivery_context)
    graph.add_node("execute_cart_action", _execute_cart_action)
    graph.add_node("zep_memory_write", _zep_memory_write)

    graph.add_edge(START, "load_zep_memory")
    graph.add_edge("load_zep_memory", "analyze_intent")
    graph.add_edge("analyze_intent", "master_flow")
    graph.add_conditional_edges(
        "master_flow",
        route_after_master_flow,
        {
            "retrieve_hybrid_context": "retrieve_hybrid_context",
            "call_mcp_tools": "call_mcp_tools",
            "run_checkout_graph": "run_checkout_graph",
            "resolve_cart_product": "resolve_cart_product",
            "resolve_delivery_context": "resolve_delivery_context",
            "generate_response": "generate_response",
        },
    )
    graph.add_edge("retrieve_hybrid_context", "resolve_delivery_context")
    graph.add_conditional_edges(
        "resolve_delivery_context",
        route_after_resolve_delivery_context,
        {
            "agent_loop": "agent_loop",
            "call_mcp_tools": "call_mcp_tools",
            "generate_response": "generate_response",
        },
    )
    graph.add_edge("agent_loop", "generate_response")
    graph.add_edge("call_mcp_tools", "generate_response")
    graph.add_edge("run_checkout_graph", "generate_response")
    graph.add_edge("resolve_cart_product", "execute_cart_action")
    graph.add_edge("execute_cart_action", "generate_response")
    graph.add_edge("generate_response", "zep_memory_write")
    graph.add_edge("zep_memory_write", END)

    return graph.compile(checkpointer=checkpointer)


async def get_shopping_graph(
    redis_client: RedisClient,
    *,
    deps: ShoppingGraphDeps | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Factory: optional Redis checkpointer + compiled shopping graph."""
    checkpointer = await get_checkpointer(redis_client)
    resolved_deps = deps or ShoppingGraphDeps()
    if resolved_deps.redis_client is None:
        resolved_deps = ShoppingGraphDeps(
            kapruka_service=resolved_deps.kapruka_service,
            client_ip=resolved_deps.client_ip,
            genai_client=resolved_deps.genai_client,
            neo4j_client=resolved_deps.neo4j_client,
            zep_client=resolved_deps.zep_client,
            redis_client=redis_client,
        )
    return build_shopping_graph(checkpointer=checkpointer, deps=resolved_deps)


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


def _per_turn_agent_reset_fields() -> dict[str, Any]:
    """Fields cleared on each follow-up turn so prior agent-loop state cannot leak."""
    return {
        "tool_trace": [],
        "tool_results": {},
        "tool_call_count": 0,
        "agent_clarifying_question": None,
        "master_clarifying_question": None,
        "master_flow_invoked": None,
        "master_flow_decision": None,
        "master_flow_mismatch_reason": None,
        "active_flow": None,
        "checkout_paused": None,
        "agent_tool_error": None,
        "agent_loop_done": None,
        "agent_loop_exit_reason": None,
        "agent_loop_iterations": None,
        "delivery_city_raw": None,
        "delivery_city_canonical": None,
        "delivery_city_status": None,
        "delivery_city_candidates": None,
        "delivery_context_ready": None,
    }


def append_message_state(message: str, *, currency: str | None = None) -> AgentState:
    """Delta state for a follow-up turn; checkpoint carries conversation context only."""
    delta: dict[str, Any] = {
        "messages": [HumanMessage(content=message)],
        **_per_turn_agent_reset_fields(),
    }
    if currency is not None:
        delta["currency"] = currency
    return cast(AgentState, delta)
