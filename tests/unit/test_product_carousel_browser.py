"""Browser verification for product carousel mobile layout."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright

from app.templating import render_product_carousel

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PRODUCTS_DIR = FIXTURES_DIR / "products"
APP_CSS = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "app.css"
LAZY_IMAGE_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "lazy-image.js"
).read_text(encoding="utf-8")
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _load_products() -> list[dict[str, object]]:
    names = (
        "sample_in_stock.json",
        "sample_out_of_stock.json",
        "sample_vanilla_cake.json",
    )
    return [json.loads((PRODUCTS_DIR / name).read_text(encoding="utf-8")) for name in names]


def _many_products() -> list[dict[str, object]]:
    """Repeat fixtures so the grid has enough cards for lazy-load testing."""
    # First two carousel slots render eager images; need 14+ cards for 12+ lazy slots.
    return _load_products() * 5


def _carousel_harness_html(carousel_html: str) -> str:
    css = APP_CSS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
{css}
      body {{
        width: 375px;
        max-width: 375px;
        overflow-y: hidden;
      }}
      [data-testid="product-carousel"] {{
        width: 100%;
        max-width: 343px;
      }}
    </style>
    <script>{LAZY_IMAGE_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body class="bg-canvas p-4">
    <div class="mx-auto max-w-[85%] rounded-xl border border-surface-muted bg-canvas p-3">
      {carousel_html}
    </div>
  </body>
</html>"""


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function(
        "() => window.Alpine && document.querySelector('[x-data]')?._x_dataStack"
    )


@pytest.mark.browser
def test_product_carousel_no_horizontal_overflow_on_mobile_viewport() -> None:
    """Grid stacks cards in a single column without widening the page on mobile."""
    carousel_html = render_product_carousel(_many_products())

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 375, "height": 812})
        page.set_content(_carousel_harness_html(carousel_html))
        _wait_for_alpine(page)
        page.wait_for_timeout(150)

        layout = page.evaluate(
            """() => {
              const cards = document.querySelectorAll('[data-testid="product-card"]');
              const doc = document.documentElement;
              return {
                cardCount: cards.length,
                pageOverflow: doc.scrollWidth > doc.clientWidth,
              };
            }"""
        )

        browser.close()

    assert layout["cardCount"] >= 12
    assert layout["pageOverflow"] is False


@pytest.mark.browser
def test_lazy_images_defer_load_until_scrolled_into_view() -> None:
    """Off-screen grid images load and fade in after scrolling into view."""
    carousel_html = render_product_carousel(_many_products())

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 375, "height": 400})

        def _stub_images(route, request) -> None:
            if request.resource_type == "image":
                time.sleep(0.1)
                route.fulfill(body=TINY_PNG, content_type="image/png")
                return
            route.continue_()

        page.route("**/*", _stub_images)
        page.set_content(_carousel_harness_html(carousel_html))
        _wait_for_alpine(page)

        off_screen = page.evaluate(
            """() => {
              const lazyImages = [...document.querySelectorAll('[data-testid="lazy-image"]')];
              const bottom = lazyImages.reduce((best, el) => {
                const top = el.getBoundingClientRect().top;
                return top > best.top ? { el, top } : best;
              }, { el: lazyImages[0], top: -1 });
              const target = bottom.el;
              const rect = target.getBoundingClientRect();
              const offScreen = rect.top >= window.innerHeight;
              const data = target._x_dataStack?.[0];
              return {
                offScreen,
                inView: data?.inView ?? false,
                loaded: data?.loaded ?? false,
                hasImg: Boolean(target.querySelector('img')),
                lazyCount: lazyImages.length,
              };
            }"""
        )
        assert off_screen["lazyCount"] >= 12
        assert off_screen["offScreen"] is True
        assert off_screen["inView"] is False
        assert off_screen["hasImg"] is False

        page.evaluate(
            """() => {
              const lazyImages = [...document.querySelectorAll('[data-testid="lazy-image"]')];
              const bottom = lazyImages.reduce((best, el) => {
                const top = el.getBoundingClientRect().top;
                return top > best.top ? { el, top } : best;
              }, { el: lazyImages[0], top: -1 });
              bottom.el.scrollIntoView({ block: 'center' });
            }"""
        )
        page.wait_for_timeout(100)

        page.wait_for_function(
            """() => {
              const lazyImages = [...document.querySelectorAll('[data-testid="lazy-image"]')];
              const bottom = lazyImages.reduce((best, el) => {
                const top = el.getBoundingClientRect().top;
                return top > best.top ? { el, top } : best;
              }, { el: lazyImages[0], top: -1 });
              const target = bottom.el;
              const data = target._x_dataStack?.[0];
              const img = target.querySelector('img');
              return Boolean(data?.inView && img);
            }"""
        )

        page.wait_for_function(
            """() => {
              const lazyImages = [...document.querySelectorAll('[data-testid="lazy-image"]')];
              const bottom = lazyImages.reduce((best, el) => {
                const top = el.getBoundingClientRect().top;
                return top > best.top ? { el, top } : best;
              }, { el: lazyImages[0], top: -1 });
              const target = bottom.el;
              const data = target._x_dataStack?.[0];
              const img = target.querySelector('img');
              return Boolean(data?.loaded && img?.classList.contains('opacity-100'));
            }"""
        )

        browser.close()
