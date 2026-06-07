"""Main shopping LangGraph — nodes wired incrementally from PRD-028 onward."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graphs.state import AgentState
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient


async def _bootstrap_turn(state: AgentState) -> dict[str, Any]:
    """Stub node until analyze_intent (PRD-028); bumps turn counter for checkpoint tests."""
    turn = state.get("tool_call_count") or 0
    return {"tool_call_count": turn + 1}


def build_shopping_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the shopping StateGraph with an optional Redis checkpointer."""
    graph = StateGraph(AgentState)
    graph.add_node("bootstrap", _bootstrap_turn)
    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", END)
    return graph.compile(checkpointer=checkpointer)


async def get_shopping_graph(
    redis_client: RedisClient,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Factory: Redis checkpointer + compiled shopping graph."""
    checkpointer = await get_checkpointer(redis_client)
    return build_shopping_graph(checkpointer=checkpointer)


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
