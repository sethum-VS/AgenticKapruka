"""Seed checkout fields from shopping-session agent state."""

from __future__ import annotations

from graphs.checkout_state import CheckoutState
from graphs.state import AgentState


def seed_checkout_from_agent_state(
    checkout: CheckoutState,
    agent: AgentState,
) -> CheckoutState:
    """Copy session delivery fields into empty checkout slots only."""
    merged = dict(checkout)

    delivery_city = merged.get("delivery_city")
    if not (isinstance(delivery_city, str) and delivery_city.strip()):
        session_city = agent.get("session_delivery_city_canonical")
        if isinstance(session_city, str) and session_city.strip():
            merged["delivery_city"] = session_city.strip()

    delivery_date = merged.get("delivery_date")
    if not (isinstance(delivery_date, str) and delivery_date.strip()):
        session_date = agent.get("session_delivery_date")
        if isinstance(session_date, str) and session_date.strip():
            merged["delivery_date"] = session_date.strip()

    delivery_address = merged.get("delivery_address")
    if not (isinstance(delivery_address, str) and delivery_address.strip()):
        raw_address = agent.get("session_shipment_address_raw")
        if isinstance(raw_address, str) and raw_address.strip():
            merged["delivery_address"] = raw_address.strip()[:250]

    return merged  # type: ignore[return-value]
