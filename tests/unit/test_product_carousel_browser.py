"""Browser verification for product carousel mobile layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright

from app.templating import render_product_carousel

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PRODUCTS_DIR = FIXTURES_DIR / "products"
APP_CSS = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "app.css"


def _load_products() -> list[dict[str, object]]:
    names = (
        "sample_in_stock.json",
        "sample_out_of_stock.json",
        "sample_vanilla_cake.json",
    )
    return [json.loads((PRODUCTS_DIR / name).read_text(encoding="utf-8")) for name in names]


def _carousel_harness_html(carousel_html: str) -> str:
    css = APP_CSS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>{css}</style>
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
    carousel_html = render_product_carousel(_load_products())

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 375, "height": 812})
        page.set_content(_carousel_harness_html(carousel_html))
        _wait_for_alpine(page)

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

    assert layout["cardCount"] >= 3
    assert layout["pageOverflow"] is False
    assert layout["trackOverflow"] is True
    assert layout["trackClientWidth"] > 0
