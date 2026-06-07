"""Structure and validation tests for delivery address form."""

from __future__ import annotations

import pytest

from app.templating import (
    _create_templates,
    render_delivery_field_error,
    render_delivery_form,
    render_delivery_form_validation_response,
)
from lib.checkout.delivery import DeliveryFormValues, parse_delivery_form


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_delivery_form_renders_all_fields_with_htmx_validation() -> None:
    """Form posts to /checkout/validate-delivery and swaps outerHTML in place."""
    html = render_delivery_form(min_date="2026-06-08")

    assert 'data-testid="delivery-form"' in html
    assert 'hx-post="/checkout/validate-delivery"' in html
    assert 'hx-target="#delivery-form"' in html
    assert 'hx-swap="outerHTML"' in html
    assert 'name="address"' in html
    assert 'name="city"' in html
    assert 'name="location_type"' in html
    assert 'name="date"' in html
    assert 'name="instructions"' in html
    assert 'min="2026-06-08"' in html
    assert '<option value="house" selected>' in html
    assert 'hx-get="/partials/delivery-cities"' in html


def test_delivery_form_preserves_submitted_values() -> None:
    """Re-rendered form keeps user-entered field values."""
    values = DeliveryFormValues(
        address="42 Lotus Road",
        city="Colombo 03",
        location_type="apartment",
        date="2026-06-10",
        instructions="Ring bell twice",
    )
    html = render_delivery_form(values=values, min_date="2026-06-08")

    assert 'value="42 Lotus Road"' in html
    assert 'value="Colombo 03"' in html
    assert '<option value="apartment" selected>' in html
    assert 'value="2026-06-10"' in html
    assert "Ring bell twice" in html


def test_parse_delivery_form_rejects_short_address() -> None:
    """Address must be 3–250 characters per Delivery Pydantic model."""
    values = DeliveryFormValues(
        address="ab",
        city="Colombo 03",
        location_type="house",
        date="2026-06-10",
    )
    delivery, errors = parse_delivery_form(values)

    assert delivery is None
    assert "address" in errors


def test_parse_delivery_form_rejects_long_instructions() -> None:
    """Instructions max 250 characters."""
    values = DeliveryFormValues(
        address="42 Lotus Road",
        city="Colombo 03",
        location_type="house",
        date="2026-06-10",
        instructions="x" * 251,
    )
    delivery, errors = parse_delivery_form(values)

    assert delivery is None
    assert "instructions" in errors


def test_parse_delivery_form_accepts_valid_payload() -> None:
    """Valid fields parse into Kapruka Delivery model."""
    values = DeliveryFormValues(
        address="42 Lotus Road",
        city="Colombo 03",
        location_type="office",
        date="2026-06-10",
        instructions="Leave at reception",
    )
    delivery, errors = parse_delivery_form(values)

    assert errors == {}
    assert delivery is not None
    assert delivery.address == "42 Lotus Road"
    assert delivery.city == "Colombo 03"
    assert delivery.location_type == "office"
    assert delivery.date == "2026-06-10"
    assert delivery.instructions == "Leave at reception"


def test_delivery_field_error_oob_fragment() -> None:
    """Field errors render as HTMX OOB swap fragments."""
    html = render_delivery_field_error(
        field="address",
        message="String should have at least 3 characters",
    )

    assert 'id="delivery-address-error"' in html
    assert 'hx-swap-oob="innerHTML"' in html
    assert 'data-testid="delivery-address-error"' in html
    assert 'role="alert"' in html
    assert "String should have at least 3 characters" in html


def test_delivery_form_validation_response_appends_oob_errors() -> None:
    """Validation response includes form HTML plus OOB error fragments."""
    values = DeliveryFormValues(address="ab", city="Galle", date="2026-06-10")
    html = render_delivery_form_validation_response(
        values=values,
        errors={"address": "String should have at least 3 characters"},
    )

    assert 'data-testid="delivery-form"' in html
    assert 'value="Galle"' in html
    assert 'hx-swap-oob="innerHTML"' in html
    assert 'data-testid="delivery-address-error"' in html
