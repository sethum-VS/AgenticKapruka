"""Checkout sub-graph step nodes with validation gates."""

from __future__ import annotations

import re
from typing import Any, cast

from graphs.checkout_state import (
    CHECKOUT_STEP_ORDER,
    CheckoutState,
    all_steps_before_review_valid,
    next_checkout_step,
    prev_checkout_step,
    resolve_navigation,
)
from graphs.state import CheckoutStep
from lib.checkout.delivery import DeliveryFormValues, parse_delivery_form
from lib.checkout.order import build_create_order_from_checkout
from lib.checkout.payment import payment_cta_from_finalize
from lib.checkout.recipient import RecipientFormValues, parse_recipient_form
from lib.checkout.review import review_context_from_checkout_state
from lib.checkout.sender import SenderFormValues, parse_sender_form
from lib.kapruka.errors import KaprukaError
from lib.kapruka.service import KaprukaService
from lib.redis.cart import get_cart
from lib.redis.client import RedisClient
from lib.redis.order import store_pending_order
from lib.utils.timezone import is_past_colombo_date

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


async def validate_cart_step(
    state: CheckoutState,
    *,
    redis_client: RedisClient | None = None,
) -> tuple[bool, dict[str, str], list[dict[str, Any]]]:
    """Ensure the session cart has at least one item."""
    session_id = state.get("session_id", "")
    items: list[dict[str, Any]] = list(state.get("cart_items") or [])

    if redis_client is not None and session_id:
        cart_rows = await get_cart(redis_client, session_id)
        items = [row.model_dump() for row in cart_rows]

    if not items:
        return False, {"cart": "Add at least one item to your cart before checkout."}, items
    return True, {}, items


def validate_delivery_city_step(state: CheckoutState) -> tuple[bool, dict[str, str]]:
    """Validate delivery city field."""
    city = (state.get("delivery_city") or "").strip()
    if len(city) < 2:
        return False, {"delivery_city": "Select a deliverable city (at least 2 characters)."}
    if len(city) > 100:
        return False, {"delivery_city": "City name must be at most 100 characters."}
    return True, {}


def validate_delivery_date_step(state: CheckoutState) -> tuple[bool, dict[str, str]]:
    """Validate delivery date format and Colombo calendar constraint."""
    date_value = (state.get("delivery_date") or "").strip()
    city = (state.get("delivery_city") or "").strip()

    if not city:
        return False, {"delivery_city": "Set a delivery city before choosing a date."}
    if not _ISO_DATE.match(date_value):
        return False, {"delivery_date": "date must be YYYY-MM-DD"}
    if is_past_colombo_date(date_value):
        return False, {"delivery_date": "Delivery date cannot be in the past."}
    return True, {}


def validate_recipient_step(state: CheckoutState) -> tuple[bool, dict[str, str]]:
    """Validate recipient name and phone via Kapruka Recipient model."""
    values = RecipientFormValues(
        name=state.get("recipient_name") or "",
        phone=state.get("recipient_phone") or "",
    )
    _recipient, errors = parse_recipient_form(values)
    if errors:
        return False, errors
    return True, {}


def validate_sender_step(state: CheckoutState) -> tuple[bool, dict[str, str]]:
    """Validate sender name and anonymous flag via Kapruka Sender model."""
    values = SenderFormValues(
        name=state.get("sender_name") or "",
        anonymous=bool(state.get("sender_anonymous")),
    )
    _sender, errors = parse_sender_form(values)
    if errors:
        return False, errors
    return True, {}


def validate_review_step(state: CheckoutState) -> tuple[bool, dict[str, str]]:
    """Review requires all prior steps to be valid and full delivery payload."""
    step_valid = state.get("step_valid") or {}
    if not all_steps_before_review_valid(step_valid):
        return False, {"review": "Complete all checkout steps before review."}

    values = DeliveryFormValues(
        address=state.get("delivery_address") or "",
        city=state.get("delivery_city") or "",
        location_type=state.get("delivery_location_type") or "house",
        date=state.get("delivery_date") or "",
        instructions=state.get("delivery_instructions") or "",
    )
    _delivery, errors = parse_delivery_form(values)
    if errors:
        return False, errors
    return True, {}


async def execute_finalize_step(
    state: CheckoutState,
    *,
    redis_client: RedisClient | None = None,
    kapruka_service: KaprukaService | None = None,
    client_ip: str = "127.0.0.1",
) -> tuple[bool, dict[str, str], dict[str, Any]]:
    """Place order via Kapruka MCP and persist order_ref / expires_at in Redis."""
    step_valid = state.get("step_valid") or {}
    if not step_valid.get("review"):
        return False, {"finalize": "Confirm your order at review before placing it."}, {}
    if not all_steps_before_review_valid(step_valid):
        return False, {"finalize": "Complete all checkout steps before placing your order."}, {}

    if kapruka_service is None:
        return False, {"finalize": "Checkout is temporarily unavailable. Please try again."}, {}

    session_id = state.get("session_id", "")
    items: list[dict[str, Any]] = list(state.get("cart_items") or [])
    if redis_client is not None and session_id and not items:
        cart_rows = await get_cart(redis_client, session_id)
        items = [row.model_dump() for row in cart_rows]

    try:
        recipient, delivery, sender, cart, gift_message, currency = (
            build_create_order_from_checkout(state, items)
        )
    except ValueError as exc:
        return False, {"finalize": str(exc)}, {}

    try:
        response = await kapruka_service.create_order(
            client_ip,
            cart=cart,
            recipient=recipient,
            delivery=delivery,
            sender=sender,
            gift_message=gift_message,
            currency=currency,
        )
    except KaprukaError as exc:
        return False, {"finalize": exc.message}, {}

    extra: dict[str, Any] = {
        "checkout_url": response.checkout_url,
        "order_ref": response.order_ref,
        "expires_at": response.expires_at,
        "order_summary": response.summary.model_dump(),
    }

    if redis_client is not None and session_id:
        await store_pending_order(
            redis_client,
            session_id,
            order_ref=response.order_ref,
            expires_at=response.expires_at,
        )

    return True, {}, extra


async def _validate_step(
    step: CheckoutStep,
    state: CheckoutState,
    *,
    redis_client: RedisClient | None = None,
    kapruka_service: KaprukaService | None = None,
    client_ip: str = "127.0.0.1",
) -> tuple[bool, dict[str, str], dict[str, Any]]:
    """Run step validation; returns (ok, errors, extra_state_fields)."""
    extra: dict[str, Any] = {}

    if step == "cart":
        ok, errors, items = await validate_cart_step(state, redis_client=redis_client)
        extra["cart_items"] = items
        return ok, errors, extra

    if step == "delivery_city":
        ok, errors = validate_delivery_city_step(state)
        if ok and kapruka_service is not None:
            city = (state.get("delivery_city") or "").strip()
            cities = await kapruka_service.list_delivery_cities(
                client_ip,
                query=city,
                limit=50,
            )
            if city not in cities:
                msg = "City is not available for Kapruka delivery."
                return False, {"delivery_city": msg}, extra
        return ok, errors, extra

    if step == "delivery_date":
        ok, errors = validate_delivery_date_step(state)
        if ok and kapruka_service is not None:
            city = (state.get("delivery_city") or "").strip()
            date_value = (state.get("delivery_date") or "").strip()
            result = await kapruka_service.check_delivery(
                client_ip,
                city=city,
                delivery_date=date_value,
            )
            if not result.available:
                reason = result.reason or "Selected date is not available for delivery."
                return False, {"delivery_date": reason}, extra
        return ok, errors, extra

    if step == "recipient":
        ok, errors = validate_recipient_step(state)
        return ok, errors, extra

    if step == "sender":
        ok, errors = validate_sender_step(state)
        return ok, errors, extra

    if step == "review":
        ok, errors = validate_review_step(state)
        return ok, errors, extra

    if step == "finalize":
        return await execute_finalize_step(
            state,
            redis_client=redis_client,
            kapruka_service=kapruka_service,
            client_ip=client_ip,
        )

    return False, {"current_step": f"Unknown checkout step: {step}"}, extra


async def process_checkout_step(
    step: CheckoutStep,
    state: CheckoutState,
    *,
    redis_client: RedisClient | None = None,
    kapruka_service: KaprukaService | None = None,
    client_ip: str = "127.0.0.1",
) -> dict[str, Any]:
    """Execute one checkout step: validate, apply navigation rules, return state delta."""
    current = state.get("current_step") or "cart"
    if current != step:
        return {
            "current_step": current,
            "validation_errors": {"current_step": f"Expected step {step}, got {current}."},
        }

    action = state.get("action")
    target = state.get("target_step")
    resolved_target, nav_allowed = resolve_navigation(
        current=current,
        target=target,
        action=action,
    )

    if not nav_allowed:
        return {
            "current_step": current,
            "validation_errors": {
                "navigation": (
                    f"Cannot skip from {current} to {target}. "
                    "Complete each step in order or use back to return."
                ),
            },
            "step_valid": dict(state.get("step_valid") or {}),
        }

    step_valid = dict(state.get("step_valid") or {})

    if action == "back" and resolved_target != current:
        prev = prev_checkout_step(current)
        if prev is None or step_index_safe(resolved_target) > step_index_safe(current):
            return {
                "current_step": current,
                "validation_errors": {"navigation": "Invalid back navigation."},
                "step_valid": step_valid,
            }
        return {
            "current_step": resolved_target,
            "validation_errors": None,
            "step_valid": step_valid,
        }

    ok, errors, extra = await _validate_step(
        step,
        state,
        redis_client=redis_client,
        kapruka_service=kapruka_service,
        client_ip=client_ip,
    )

    if not ok:
        return {
            "current_step": current,
            "validation_errors": errors,
            "step_valid": step_valid,
            **extra,
        }

    step_valid[step] = True
    updates: dict[str, Any] = {
        "validation_errors": None,
        "step_valid": step_valid,
        **extra,
    }

    if step == "review":
        merged_state = dict(state)
        merged_state.update(updates)
        merged_state.update(extra)
        review_context = review_context_from_checkout_state(cast(CheckoutState, merged_state))
        if review_context is not None:
            from app.templating import render_checkout_review

            updates["response_html"] = render_checkout_review(review=review_context)

    if step == "finalize":
        payment_context = payment_cta_from_finalize(
            checkout_url=str(extra.get("checkout_url") or ""),
            order_ref=str(extra.get("order_ref") or ""),
            order_summary=extra.get("order_summary")
            if isinstance(extra.get("order_summary"), dict)
            else None,
            expires_at=str(extra.get("expires_at") or ""),
            currency=str(state.get("currency") or "LKR"),
        )
        if payment_context is not None:
            from app.templating import render_payment_cta

            updates["response_html"] = render_payment_cta(payment=payment_context)

    if action == "advance":
        nxt = next_checkout_step(current)
        if nxt is None:
            updates["current_step"] = current
        else:
            updates["current_step"] = nxt
    else:
        updates["current_step"] = current

    return updates


def step_index_safe(step: CheckoutStep) -> int:
    """Index helper that avoids raising for unknown steps."""
    try:
        return CHECKOUT_STEP_ORDER.index(step)
    except ValueError:
        return -1


def route_checkout_entry(state: CheckoutState) -> str:
    """Route START to the node matching current_step."""
    step = state.get("current_step") or "cart"
    if step not in CHECKOUT_STEP_ORDER:
        return "cart"
    return step
