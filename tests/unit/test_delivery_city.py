"""Structure tests for delivery city debounced autocomplete."""

from __future__ import annotations

import pytest

from app.templating import (
    _create_templates,
    render_delivery_city,
    render_delivery_city_suggestions,
)


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_delivery_city_input_htmx_debounced_autocomplete() -> None:
    """City input issues debounced hx-get to /partials/delivery-cities."""
    html = render_delivery_city()

    assert 'data-testid="delivery-city-field"' in html
    assert 'data-testid="delivery-city-input"' in html
    assert 'id="delivery-city"' in html
    assert 'name="q"' in html
    assert 'hx-get="/partials/delivery-cities"' in html
    assert 'hx-trigger="keyup changed delay:300ms"' in html
    assert 'hx-target="#delivery-city-suggestions"' in html
    assert 'hx-swap="innerHTML"' in html
    assert 'id="delivery-city-suggestions"' in html
    assert 'role="listbox"' in html


def test_delivery_city_suggestions_render_li_items() -> None:
    """Suggestion partial renders li rows with city names."""
    html = render_delivery_city_suggestions(
        cities=["Colombo 03", "Colombo 07", "Galle"],
    )

    assert html.count('data-testid="delivery-city-suggestion"') == 3
    assert 'data-city="Colombo 03"' in html
    assert ">Colombo 03<" in html
    assert ">Colombo 07<" in html
    assert ">Galle<" in html
    assert 'role="option"' in html


def test_delivery_city_suggestions_empty_for_no_matches() -> None:
    """Empty city list renders no suggestion items."""
    html = render_delivery_city_suggestions(cities=[])

    assert html.strip() == ""
