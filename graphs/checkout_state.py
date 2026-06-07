"""Checkout sub-graph state schema and deterministic step navigation."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from graphs.state import CheckoutStep, CurrencyCode

CheckoutAction = Literal["advance", "back"]

CHECKOUT_STEP_ORDER: tuple[CheckoutStep, ...] = (
    "cart",
    "delivery_city",
    "delivery_date",
    "recipient",
    "sender",
    "review",
)


class CheckoutState(TypedDict, total=False):
    """State for the checkout LangGraph sub-graph."""

    session_id: str
    currency: CurrencyCode | None
    current_step: CheckoutStep
    target_step: CheckoutStep | None
    action: CheckoutAction | None

    cart_items: list[dict[str, Any]]
    delivery_city: str | None
    delivery_address: str | None
    delivery_location_type: str | None
    delivery_date: str | None
    delivery_instructions: str | None

    recipient_name: str | None
    recipient_phone: str | None

    sender_name: str | None
    sender_anonymous: bool | None

    gift_message: str | None

    step_valid: dict[str, bool]
    validation_errors: dict[str, str] | None
    response_html: str | None


def step_index(step: CheckoutStep) -> int:
    """Return the ordered index of a checkout step."""
    return CHECKOUT_STEP_ORDER.index(step)


def next_checkout_step(step: CheckoutStep) -> CheckoutStep | None:
    """Return the step after ``step``, or None when already at review."""
    idx = step_index(step)
    if idx >= len(CHECKOUT_STEP_ORDER) - 1:
        return None
    return CHECKOUT_STEP_ORDER[idx + 1]


def prev_checkout_step(step: CheckoutStep) -> CheckoutStep | None:
    """Return the step before ``step``, or None when already at cart."""
    idx = step_index(step)
    if idx <= 0:
        return None
    return CHECKOUT_STEP_ORDER[idx - 1]


def resolve_navigation(
    *,
    current: CheckoutStep,
    target: CheckoutStep | None,
    action: CheckoutAction | None,
) -> tuple[CheckoutStep, bool]:
    """Resolve requested navigation without allowing forward skips.

    Returns (resolved_step, allowed). ``allowed`` is False when a skip was rejected.
    """
    if target is None:
        return current, True

    cur_idx = step_index(current)
    tgt_idx = step_index(target)

    if action == "back":
        if tgt_idx < cur_idx:
            return target, True
        return current, tgt_idx == cur_idx

    if action == "advance":
        if tgt_idx == cur_idx:
            return current, True
        if tgt_idx == cur_idx + 1:
            return target, True
        return current, False

    if tgt_idx > cur_idx + 1:
        return current, False
    if tgt_idx <= cur_idx:
        return target, True
    return current, False


def initial_checkout_state(
    *,
    session_id: str,
    currency: CurrencyCode | None = None,
    cart_items: list[dict[str, Any]] | None = None,
) -> CheckoutState:
    """Build initial checkout state starting at the cart step."""
    state: dict[str, Any] = {
        "session_id": session_id,
        "current_step": "cart",
        "step_valid": {},
        "cart_items": cart_items or [],
    }
    if currency is not None:
        state["currency"] = currency
    return cast(CheckoutState, state)


def all_steps_before_review_valid(step_valid: dict[str, bool] | None) -> bool:
    """True when every pre-review step has passed validation."""
    if not step_valid:
        return False
    return all(step_valid.get(step) for step in CHECKOUT_STEP_ORDER[:-1])
