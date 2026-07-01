"""Unit tests for checkout session prefill from agent state."""

from __future__ import annotations

from graphs.checkout_state import initial_checkout_state
from graphs.state import AgentState
from lib.checkout.prefill import seed_checkout_from_agent_state

_SESSION_ID = "sess-prefill-001"


def test_seed_checkout_fills_empty_delivery_fields_from_session() -> None:
    checkout = initial_checkout_state(session_id=_SESSION_ID)
    agent: AgentState = {
        "session_delivery_city_canonical": "Colombo 03",
        "session_delivery_date": "2026-06-28",
        "session_shipment_address_raw": "123 Galle Road",
    }

    seeded = seed_checkout_from_agent_state(checkout, agent)

    assert seeded["delivery_city"] == "Colombo 03"
    assert seeded["delivery_date"] == "2026-06-28"
    assert seeded["delivery_address"] == "123 Galle Road"


def test_seed_checkout_does_not_overwrite_persisted_checkout_fields() -> None:
    checkout = initial_checkout_state(session_id=_SESSION_ID)
    checkout["delivery_city"] = "Galle"
    checkout["delivery_date"] = "2026-06-20"
    agent: AgentState = {
        "session_delivery_city_canonical": "Colombo 03",
        "session_delivery_date": "2026-06-28",
    }

    seeded = seed_checkout_from_agent_state(checkout, agent)

    assert seeded["delivery_city"] == "Galle"
    assert seeded["delivery_date"] == "2026-06-20"
