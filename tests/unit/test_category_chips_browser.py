"""Browser verification for category chip HTMX filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, Response, Route, sync_playwright

from app.templating import render_category_chips

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CATEGORIES_FIXTURE = FIXTURES_DIR / "hybrid_context" / "sample_categories.json"
APP_CSS = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "app.css"


def _load_categories() -> list[dict[str, object]]:
    return json.loads(CATEGORIES_FIXTURE.read_text(encoding="utf-8"))


def _chips_harness_html(chips_html: str) -> str:
    css = APP_CSS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <base href="http://localhost/" />
    <style>{css}</style>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  </head>
  <body class="bg-commerce-cream p-4">
    {chips_html}
    <div id="results" data-testid="search-results">All products</div>
  </body>
</html>"""


def _wait_for_htmx(page: Page) -> None:
    page.wait_for_function("() => window.htmx")


@pytest.mark.browser
def test_category_chip_click_triggers_htmx_without_full_reload() -> None:
    """Clicking a chip fetches the search partial into #results without navigation."""
    chips_html = render_category_chips(_load_categories(), active_category="Flowers")
    requests: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        def handle_search(route: Route) -> None:
            if "/partials/search" not in route.request.url:
                route.continue_()
                return
            requests.append(route.request.url)
            route.fulfill(
                status=200,
                headers={"Content-Type": "text/html"},
                body="<div data-testid='filtered-results'>Cakes only</div>",
            )

        page.route("**/*", handle_search)
        page.set_content(_chips_harness_html(chips_html))
        _wait_for_htmx(page)
        page.evaluate("() => window.htmx.process(document.body)")
        page.wait_for_function(
            "() => document.querySelector('[data-category=\"Cakes\"]')?.hasAttribute('hx-get')"
        )

        def _is_search_response(response: Response) -> bool:
            return "/partials/search" in response.url

        with page.expect_response(_is_search_response) as response_info:
            page.locator('[data-category="Cakes"]').click()
        assert "category=Cakes" in response_info.value.url

        page.wait_for_selector('[data-testid="filtered-results"]')

        stayed_on_harness = page.evaluate(
            """() => ({
              hasResults: !!document.querySelector('[data-testid="search-results"]'),
              filteredText: document.querySelector('[data-testid="filtered-results"]')?.textContent,
            })"""
        )

        browser.close()

    assert any("partials/search?category=Cakes" in url for url in requests)
    assert stayed_on_harness["hasResults"] is True
    assert stayed_on_harness["filteredText"] == "Cakes only"
