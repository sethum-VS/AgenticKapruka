"""Structure tests for templates/components/category_chips.html."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.templating import _create_templates, categories_for_chips, render_category_chips

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CATEGORIES_FIXTURE = FIXTURES_DIR / "hybrid_context" / "sample_categories.json"


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _load_categories() -> list[dict[str, object]]:
    return json.loads(CATEGORIES_FIXTURE.read_text(encoding="utf-8"))


def test_category_chips_render_htmx_search_links() -> None:
    """Each chip issues hx-get to /partials/search and swaps into #results."""
    html = render_category_chips(_load_categories())

    assert 'data-testid="category-chips"' in html
    assert html.count('data-testid="category-chip"') == 3
    assert 'hx-target="#results"' in html
    assert 'hx-swap="innerHTML"' in html
    assert 'hx-trigger="click"' in html
    assert 'hx-get="/partials/search?category=Flowers"' in html
    assert 'hx-get="/partials/search?category=Cakes"' in html
    assert 'hx-get="/partials/search?category=Gifts"' in html
    assert 'aria-label="Filter by category"' in html


def test_category_chips_highlight_active_variant() -> None:
    """Active chip uses chip-active; inactive chips use chip-filter."""
    html = render_category_chips(_load_categories(), active_category="Cakes")

    assert 'data-category="Cakes"' in html
    assert 'class="chip-active"' in html
    assert html.count('class="chip-filter"') == 2
    assert 'aria-pressed="true"' in html
    assert html.count('aria-pressed="false"') == 2


def test_category_chips_urlencodes_category_names() -> None:
    """Category names with spaces are URL-encoded in hx-get."""
    categories = [{"display_name": "Gift Hampers"}]
    html = render_category_chips(categories)

    assert 'hx-get="/partials/search?category=Gift%20Hampers"' in html


def test_category_chips_renders_empty_for_no_categories() -> None:
    """Empty category list renders nothing."""
    html = render_category_chips([])

    assert html.strip() == ""


def test_categories_for_chips_deduplicates_vector_and_traversal() -> None:
    """hybrid_context vector_hits and categories merge into unique chip rows."""
    hybrid_context = {
        "vector_hits": [
            {"id": "category:flowers", "display_name": "Flowers", "score": 0.9},
            {"id": "category:cakes", "display_name": "Cakes", "score": 0.8},
        ],
        "categories": [
            {"id": "category:flowers", "display_name": "Flowers", "hop": 0},
            {"id": "category:gifts", "display_name": "Gifts", "hop": 1},
        ],
    }

    chips = categories_for_chips(hybrid_context)

    assert [chip["display_name"] for chip in chips] == ["Flowers", "Cakes", "Gifts"]


def test_categories_for_chips_empty_context() -> None:
    """Missing or empty hybrid_context yields no chips."""
    assert categories_for_chips(None) == []
    assert categories_for_chips({}) == []


@pytest.mark.asyncio
async def test_partials_search_returns_results_fragment() -> None:
    """GET /partials/search returns HTML fragment for HTMX #results swap."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/partials/search", params={"category": "Cakes"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-testid="search-results"' in response.text
    assert 'data-category="Cakes"' in response.text
    assert "Showing Cakes" in response.text
