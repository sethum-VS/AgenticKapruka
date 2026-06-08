"""Checkout LangGraph sub-graph — deterministic step state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graphs.checkout_state import CHECKOUT_STEP_ORDER, CheckoutState
from graphs.nodes.checkout_steps import process_checkout_step, route_checkout_entry
from graphs.state import CheckoutStep
from lib.kapruka.service import KaprukaService
from lib.redis.client import RedisClient


@dataclass(frozen=True, slots=True)
class CheckoutGraphDeps:
    """Injectable dependencies for checkout graph nodes."""

    redis_client: RedisClient | None = None
    kapruka_service: KaprukaService | None = None
    client_ip: str = "127.0.0.1"


def _make_step_node(step: CheckoutStep, deps: CheckoutGraphDeps) -> Any:
    async def _node(state: CheckoutState) -> dict[str, Any]:
        return await process_checkout_step(
            step,
            state,
            redis_client=deps.redis_client,
            kapruka_service=deps.kapruka_service,
            client_ip=deps.client_ip,
        )

    return _node


def build_checkout_graph(
    *,
    deps: CheckoutGraphDeps | None = None,
) -> CompiledStateGraph[CheckoutState, None, CheckoutState, CheckoutState]:
    """Compile the checkout StateGraph with one node per step."""
    resolved = deps or CheckoutGraphDeps()
    graph = StateGraph(CheckoutState)

    for step in CHECKOUT_STEP_ORDER:
        graph.add_node(step, _make_step_node(step, resolved))

    graph.add_conditional_edges(
        START,
        route_checkout_entry,
        {step: step for step in CHECKOUT_STEP_ORDER},
    )

    for step in CHECKOUT_STEP_ORDER:
        graph.add_edge(step, END)

    return graph.compile()


def get_checkout_graph(
    *,
    deps: CheckoutGraphDeps | None = None,
) -> CompiledStateGraph[CheckoutState, None, CheckoutState, CheckoutState]:
    """Factory for the compiled checkout sub-graph (no checkpointer — session scoped)."""
    return build_checkout_graph(deps=deps)
