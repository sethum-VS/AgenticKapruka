"""Structure and layout tests for templates/components/product_carousel.html."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.templating import _create_templates, render_product_carousel

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PRODUCTS_DIR = FIXTURES_DIR / "products"


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _load_product_fixture(name: str) -> dict[str, object]:
    return json.loads((PRODUCTS_DIR / name).read_text(encoding="utf-8"))


def _carousel_products() -> list[dict[str, object]]:
    return [
        _load_product_fixture("sample_in_stock.json"),
        _load_product_fixture("sample_out_of_stock.json"),
        _load_product_fixture("sample_vanilla_cake.json"),
    ]


def test_product_carousel_renders_responsive_grid() -> None:
    """Carousel uses a responsive 2-column grid for product cards in chat."""
    html = render_product_carousel(_carousel_products())

    assert 'data-testid="product-carousel"' in html
    assert 'data-testid="product-carousel-track"' in html
    assert "grid" in html
    assert "grid-cols-1" in html
    assert "md:grid-cols-2" in html
    assert html.count('data-testid="product-card"') == 3


def test_product_carousel_includes_region_semantics() -> None:
    """Product region is keyboard-focusable with an accessible label."""
    html = render_product_carousel(_carousel_products())

    assert 'role="region"' in html
    assert 'aria-label="Product carousel"' in html
    assert 'tabindex="0"' in html


def test_product_carousel_mobile_overflow_containment_classes() -> None:
    """Outer container constrains width on narrow viewports."""
    html = render_product_carousel(_carousel_products())

    assert 'class="relative w-full min-w-0"' in html
    assert "min-w-0" in html


def test_product_carousel_renders_empty_for_no_products() -> None:
    """Empty product list renders nothing (no orphan carousel chrome)."""
    html = render_product_carousel([])

    assert html.strip() == ""


def test_product_carousel_enables_lazy_image_on_cards() -> None:
    """First two carousel cards load eagerly; remaining cards lazy-load."""
    html = render_product_carousel(_carousel_products())

    assert html.count('data-testid="lazy-image"') == 1
    assert html.count('data-testid="lazy-image-skeleton"') == 1
    assert html.count('loading="lazy"') == 2
    assert 'x-data="lazyImage"' in html
    assert 'template x-if="inView"' in html
    assert "transition-opacity duration-300" in html


def test_product_carousel_includes_product_card_fields() -> None:
    """Each embedded product_card exposes name, price, and add-to-cart HTMX."""
    html = render_product_carousel(_carousel_products())

    assert "Chocolate Fudge Birthday Cake" in html
    assert "Red Rose Bouquet" in html
    assert "Vanilla Celebration Cake" in html
    assert "Rs. 4,500" in html
    assert "Rs. 3,200" in html
    assert "Rs. 5,200" in html
    assert html.count('hx-post="/cart/add"') == 3
