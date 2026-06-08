"""Structure and validation tests for recipient form."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.templating import (
    _create_templates,
    render_recipient_field_error,
    render_recipient_form,
    render_recipient_form_validation_response,
)
from lib.checkout.recipient import RecipientFormValues, parse_recipient_form
from lib.kapruka.types import Recipient


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_recipient_form_renders_fields_with_htmx_validation() -> None:
    """Form posts to /checkout/validate-recipient and swaps outerHTML in place."""
    html = render_recipient_form()

    assert 'data-testid="recipient-form"' in html
    assert 'hx-post="/checkout/validate-recipient"' in html
    assert 'hx-target="#recipient-form"' in html
    assert 'hx-swap="outerHTML"' in html
    assert 'name="name"' in html
    assert 'name="phone"' in html
    assert 'maxlength="80"' in html


def test_recipient_form_preserves_submitted_values() -> None:
    """Re-rendered form keeps user-entered field values."""
    values = RecipientFormValues(name="Ada Lovelace", phone="0771234567")
    html = render_recipient_form(values=values)

    assert 'value="Ada Lovelace"' in html
    assert 'value="0771234567"' in html


@pytest.mark.parametrize(
    "phone",
    ["0771234567", "+94771234567"],
)
def test_recipient_accepts_valid_sri_lanka_phone_formats(phone: str) -> None:
    """Phone validator accepts local 077 and E.164 +9477 formats."""
    recipient = Recipient(name="Ada Lovelace", phone=phone)
    assert recipient.phone == phone


@pytest.mark.parametrize(
    "phone",
    [
        "771234567",
        "94771234567",
        "+94761234567",
        "0761234567",
        "07712345",
        "077123456789",
        "not-a-phone",
        "",
    ],
)
def test_recipient_rejects_invalid_phone_formats(phone: str) -> None:
    """Phone validator rejects numbers outside +9477 or 077 Sri Lanka mobile formats."""
    with pytest.raises(ValidationError) as exc_info:
        Recipient(name="Ada Lovelace", phone=phone)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("phone",) for error in errors)


def test_parse_recipient_form_rejects_empty_name() -> None:
    """Name must be 1–80 characters per Recipient Pydantic model."""
    values = RecipientFormValues(name="", phone="0771234567")
    recipient, errors = parse_recipient_form(values)

    assert recipient is None
    assert "name" in errors


def test_parse_recipient_form_rejects_invalid_phone() -> None:
    """Invalid phone returns field-level error via parse_recipient_form."""
    values = RecipientFormValues(name="Ada Lovelace", phone="12345")
    recipient, errors = parse_recipient_form(values)

    assert recipient is None
    assert "phone" in errors


def test_parse_recipient_form_accepts_valid_payload() -> None:
    """Valid fields parse into Kapruka Recipient model."""
    values = RecipientFormValues(name="Ada Lovelace", phone="+94771234567")
    recipient, errors = parse_recipient_form(values)

    assert errors == {}
    assert recipient is not None
    assert recipient.name == "Ada Lovelace"
    assert recipient.phone == "+94771234567"


def test_recipient_field_error_oob_fragment() -> None:
    """Field errors render as HTMX OOB swap fragments."""
    html = render_recipient_field_error(
        field="phone",
        message="Phone must be E.164 +9477XXXXXXX or local 077XXXXXXX format",
    )

    assert 'id="recipient-phone-error"' in html
    assert 'hx-swap-oob="innerHTML"' in html
    assert 'data-testid="recipient-phone-error"' in html
    assert 'role="alert"' in html
    assert "Phone must be E.164" in html


def test_recipient_form_validation_response_appends_oob_errors() -> None:
    """Validation response includes form HTML plus OOB error fragments."""
    values = RecipientFormValues(name="Ada Lovelace", phone="invalid")
    html = render_recipient_form_validation_response(
        values=values,
        errors={"phone": "Phone must be E.164 +9477XXXXXXX or local 077XXXXXXX format"},
    )

    assert 'data-testid="recipient-form"' in html
    assert 'value="Ada Lovelace"' in html
    assert 'hx-swap-oob="innerHTML"' in html
    assert 'data-testid="recipient-phone-error"' in html
