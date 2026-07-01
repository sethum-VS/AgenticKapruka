"""Shopping graph routing with master_flow gate."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from graphs.state import AgentState
from lib.chat.routing import route_after_master_flow


def _state(**kwargs: object) -> AgentState:
    base: dict[str, object] = {
        "messages": [HumanMessage(content="birthday cake")],
        "intent": "discovery",
    }
    base.update(kwargs)
    return base  # type: ignore[return-value]


def test_master_flow_clarify_skips_hybrid_context() -> None:
    state = _state(
        master_clarifying_question="Which delivery date works for you?",
        specificity_band="proceed",
    )
    assert route_after_master_flow(state) == "generate_response"


def test_master_flow_context_reset_still_routes_discovery() -> None:
    state = _state(
        intent="discovery",
        last_visible_products=None,
        session_search_query=None,
    )
    assert route_after_master_flow(state) == "retrieve_hybrid_context"
