"""Structure and validation tests for sender form."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.templating import (
    _create_templates,
    render_sender_field_error,
    render_sender_form,
    render_sender_form_validation_response,
)
from lib.checkout.sender import SenderFormValues, parse_sender_form
from lib.kapruka.types import Sender


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_sender_form_renders_name_and_anonymous_toggle_with_htmx_validation() -> None:
    """Form posts to /checkout/validate-sender with name (max 80) and anonymous toggle only."""
    html = render_sender_form()

    assert 'data-testid="sender-form"' in html
    assert 'hx-post="/checkout/validate-sender"' in html
    assert 'hx-target="#sender-form"' in html
    assert 'hx-swap="outerHTML"' in html
    assert 'name="name"' in html
    assert 'name="anonymous"' in html
    assert 'maxlength="80"' in html
    assert 'type="checkbox"' in html
    assert 'data-testid="sender-anonymous-toggle"' in html
    assert 'name="email"' not in html
    assert 'type="email"' not in html


def test_sender_form_shows_gift_card_preview_with_anonymous_alpine_binding() -> None:
    """Anonymous toggle drives Alpine preview text for gift card sender line."""
    html = render_sender_form()

    assert 'data-testid="gift-card-preview"' in html
    assert 'data-testid="gift-card-sender-preview"' in html
    assert "anonymous ? 'Anonymous'" in html
    assert 'x-model="anonymous"' in html


def test_sender_form_preserves_submitted_values() -> None:
    """Re-rendered form keeps user-entered name and anonymous checkbox state."""
    values = SenderFormValues(name="Ada Lovelace", anonymous=True)
    html = render_sender_form(values=values)

    assert 'value="Ada Lovelace"' in html
    assert "checked" in html
    assert "anonymous: true" in html


def test_sender_model_accepts_name_and_anonymous_only() -> None:
    """Sender Pydantic model exposes only name and anonymous fields."""
    fields = set(Sender.model_fields)
    assert fields == {"name", "anonymous"}


def test_sender_rejects_empty_name() -> None:
    """Name must be 1–80 characters per Sender Pydantic model."""
    with pytest.raises(ValidationError) as exc_info:
        Sender(name="", anonymous=False)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("name",) for error in errors)


def test_sender_rejects_name_over_80_chars() -> None:
    """Name longer than 80 characters is rejected."""
    with pytest.raises(ValidationError):
        Sender(name="x" * 81, anonymous=False)


def test_parse_sender_form_rejects_empty_name() -> None:
    """Empty name returns field-level error via parse_sender_form."""
    values = SenderFormValues(name="   ", anonymous=False)
    sender, errors = parse_sender_form(values)

    assert sender is None
    assert "name" in errors


def test_parse_sender_form_accepts_anonymous_sender() -> None:
    """Valid fields parse into Kapruka Sender model with anonymous flag."""
    values = SenderFormValues(name="Ada Lovelace", anonymous=True)
    sender, errors = parse_sender_form(values)

    assert errors == {}
    assert sender is not None
    assert sender.name == "Ada Lovelace"
    assert sender.anonymous is True


def test_parse_sender_form_defaults_anonymous_false() -> None:
    """Anonymous defaults to false when not checked on submit."""
    values = SenderFormValues(name="Charles Babbage", anonymous=False)
    sender, errors = parse_sender_form(values)

    assert errors == {}
    assert sender is not None
    assert sender.anonymous is False


def test_sender_field_error_oob_fragment() -> None:
    """Field errors render as HTMX OOB swap fragments."""
    html = render_sender_field_error(
        field="name",
        message="String should have at least 1 character",
    )

    assert 'id="sender-name-error"' in html
    assert 'hx-swap-oob="innerHTML"' in html
    assert 'data-testid="sender-name-error"' in html
    assert 'role="alert"' in html


def test_sender_form_validation_response_appends_oob_errors() -> None:
    """Validation response includes form HTML plus OOB error fragments."""
    values = SenderFormValues(name="", anonymous=False)
    html = render_sender_form_validation_response(
        values=values,
        errors={"name": "String should have at least 1 character"},
    )

    assert 'data-testid="sender-form"' in html
    assert 'hx-swap-oob="innerHTML"' in html
    assert 'data-testid="sender-name-error"' in html
