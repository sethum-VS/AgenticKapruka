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


def test_product_carousel_renders_horizontal_scroll_track() -> None:
    """Carousel uses flex overflow-x-auto with scroll snap for product cards."""
    html = render_product_carousel(_carousel_products())

    assert 'data-testid="product-carousel"' in html
    assert 'data-testid="product-carousel-track"' in html
    assert "flex" in html
    assert "overflow-x-auto" in html
    assert "snap-x" in html
    assert "snap-mandatory" in html
    assert "scroll-smooth" in html
    assert "overscroll-x-contain" in html
    assert html.count('data-testid="product-card"') == 3


def test_product_carousel_includes_alpine_navigation_buttons() -> None:
    """Prev/next Alpine buttons support click and keyboard arrow navigation."""
    html = render_product_carousel(_carousel_products())

    assert 'data-testid="carousel-prev"' in html
    assert 'data-testid="carousel-next"' in html
    assert 'aria-label="Previous products"' in html
    assert 'aria-label="Next products"' in html
    assert '@click="scrollCarousel(-1)"' in html
    assert '@click="scrollCarousel(1)"' in html
    assert '@keydown.arrow-left.prevent="scrollCarousel(-1)"' in html
    assert '@keydown.arrow-right.prevent="scrollCarousel(1)"' in html
    assert 'x-ref="track"' in html
    assert 'role="region"' in html
    assert 'aria-label="Product carousel"' in html


def test_product_carousel_mobile_overflow_containment_classes() -> None:
    """Outer container constrains width on narrow viewports; cards stay shrink-0."""
    html = render_product_carousel(_carousel_products())

    assert 'class="relative w-full min-w-0"' in html
    assert "snap-start" in html
    assert "shrink-0" in html
    assert "w-56" in html


def test_product_carousel_renders_empty_for_no_products() -> None:
    """Empty product list renders nothing (no orphan carousel chrome)."""
    html = render_product_carousel([])

    assert html.strip() == ""


def test_product_carousel_enables_lazy_image_on_cards() -> None:
    """Carousel product cards use Alpine lazyImage with skeleton placeholders."""
    html = render_product_carousel(_carousel_products())

    assert html.count('data-testid="lazy-image"') == 3
    assert html.count('data-testid="lazy-image-skeleton"') == 3
    assert 'x-data="lazyImage"' in html
    assert 'template x-if="inView"' in html
    assert "transition-opacity duration-300" in html


def test_product_carousel_includes_product_card_fields() -> None:
    """Each embedded product_card exposes name, price, and add-to-cart HTMX."""
    html = render_product_carousel(_carousel_products())

    assert "Chocolate Fudge Birthday Cake" in html
    assert "Red Rose Bouquet" in html
    assert "Vanilla Celebration Cake" in html
    assert "LKR 4,500" in html
    assert "LKR 3,200" in html
    assert "LKR 5,200" in html
    assert html.count('hx-post="/cart/add"') == 3
