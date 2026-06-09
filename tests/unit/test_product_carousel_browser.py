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
    """Repeat fixtures so carousel overflow is deterministic on narrow CI viewports."""
    return _load_products() * 4


def _carousel_harness_html(carousel_html: str) -> str:
    css = APP_CSS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
{css}
      /* Pin carousel width so overflow is deterministic on all CI runners. */
      [data-testid="product-carousel"] {{
        width: 240px;
        max-width: 240px;
      }}
      [data-testid="product-carousel-track"] {{
        width: 240px;
        max-width: 240px;
      }}
      [data-testid="product-carousel-track"] > .snap-start {{
        flex-shrink: 0;
      }}
      [data-testid="product-card"] {{
        width: 14rem;
        flex-shrink: 0;
      }}
    </style>
    <script>{LAZY_IMAGE_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body class="bg-commerce-cream p-4">
    <div class="mx-auto max-w-[85%] rounded-2xl border border-commerce-parchment bg-white p-3">
      {carousel_html}
    </div>
  </body>
</html>"""


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function(
        "() => window.Alpine && document.querySelector('[x-data]')?._x_dataStack"
    )


@pytest.mark.browser
def test_product_carousel_no_page_overflow_on_mobile_viewport() -> None:
    """Three or more cards render inside the carousel without widening the page."""
    carousel_html = render_product_carousel(_many_products())

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 375, "height": 812})
        page.set_content(_carousel_harness_html(carousel_html))
        _wait_for_alpine(page)
        page.wait_for_function(
            """() => {
              const track = document.querySelector('[data-testid="product-carousel-track"]');
              return track && track.scrollWidth > track.clientWidth + 8;
            }"""
        )

        layout = page.evaluate(
            """() => {
              const cards = document.querySelectorAll('[data-testid="product-card"]');
              const track = document.querySelector('[data-testid="product-carousel-track"]');
              const doc = document.documentElement;
              return {
                cardCount: cards.length,
                pageOverflow: doc.scrollWidth > doc.clientWidth,
                trackOverflow: track ? track.scrollWidth > track.clientWidth : false,
                trackClientWidth: track ? track.clientWidth : 0,
              };
            }"""
        )

        browser.close()

    assert layout["cardCount"] >= 12
    assert layout["pageOverflow"] is False
    assert layout["trackOverflow"] is True
    assert layout["trackClientWidth"] > 0


@pytest.mark.browser
def test_lazy_images_defer_load_until_carousel_scroll() -> None:
    """Off-screen carousel images load and fade in after scrolling into view."""
    carousel_html = render_product_carousel(_many_products())

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 375, "height": 812})

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
              const track = document.querySelector('[data-testid="product-carousel-track"]');
              const last = lazyImages[lazyImages.length - 1];
              const rect = last.getBoundingClientRect();
              const trackRect = track.getBoundingClientRect();
              const offScreen = rect.left >= trackRect.right - 4;
              const data = last._x_dataStack?.[0];
              return {
                offScreen,
                inView: data?.inView ?? false,
                loaded: data?.loaded ?? false,
                hasImg: Boolean(last.querySelector('img')),
              };
            }"""
        )
        assert off_screen["offScreen"] is True
        assert off_screen["inView"] is False
        assert off_screen["hasImg"] is False

        page.evaluate(
            """() => {
              const track = document.querySelector('[data-testid="product-carousel-track"]');
              track.scrollLeft = track.scrollWidth;
            }"""
        )
        page.wait_for_timeout(50)

        page.wait_for_function(
            """() => {
              const lazyImages = [...document.querySelectorAll('[data-testid="lazy-image"]')];
              const last = lazyImages[lazyImages.length - 1];
              const data = last._x_dataStack?.[0];
              const img = last.querySelector('img');
              return Boolean(data?.inView && img);
            }"""
        )

        page.wait_for_function(
            """() => {
              const lazyImages = [...document.querySelectorAll('[data-testid="lazy-image"]')];
              const last = lazyImages[lazyImages.length - 1];
              const data = last._x_dataStack?.[0];
              const img = last.querySelector('img');
              return Boolean(data?.loaded && img?.classList.contains('opacity-100'));
            }"""
        )

        browser.close()
