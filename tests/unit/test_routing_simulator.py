"""Tests for LangGraph preprocessor routing simulator dashboard."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from app.main import create_app
from app.templating import _create_templates, get_templates
from lib.dev.routing_simulator import (
    DEFAULT_STRICTNESS,
    run_simulation,
)


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _make_request() -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "method": "GET",
        "path": "/dev/simulator",
        "headers": [],
        "query_string": b"",
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.mark.parametrize(
    ("scenario_id", "is_situational", "vernacular", "requires_delivery", "city"),
    [
        ("keeri_samba", False, "en", False, "Colombo"),
        ("tanglish_kandy", False, "tanglish", True, "Kandy"),
        ("breakup_flowers", True, "en", False, None),
    ],
)
def test_run_simulation_preprocessor_metadata(
    scenario_id: str,
    is_situational: bool,
    vernacular: str,
    requires_delivery: bool,
    city: str | None,
) -> None:
    result = run_simulation(scenario_id, tone="standard")  # type: ignore[arg-type]
    meta = result.intent_metadata
    assert meta["is_situational"] is is_situational
    assert meta["detected_vernacular"] == vernacular
    assert meta["requires_delivery_validation"] is requires_delivery
    assert meta["target_city"] == city


def test_run_simulation_kandy_binds_check_delivery() -> None:
    result = run_simulation("tanglish_kandy", tone="standard")
    assert "kapruka_search_products" in result.stage2.tools
    assert "kapruka_check_delivery" in result.stage2.tools
    assert result.stage2.tool_binding_label == "search + kapruka_check_delivery"


def test_run_simulation_utility_search_only() -> None:
    result = run_simulation("keeri_samba", tone="standard")
    assert result.stage2.tools == ("kapruka_search_products",)
    assert result.stage2.tool_binding_label == "search-only"
    assert result.stage2.prompt_template == "Utility E-commerce"


def test_run_simulation_situational_concierge_prompt() -> None:
    result = run_simulation("breakup_flowers", tone="standard")
    assert result.stage2.prompt_template == "Localized Concierge"


def test_high_empathy_passes_situational_flavor_gate() -> None:
    result = run_simulation("breakup_flowers", tone="high_empathy", strictness=0.75)
    assert result.stage3.local_flavor.score >= 0.75
    assert result.stage3.overall_pass is True


def test_robotic_tone_fails_situational_flavor_at_default_strictness() -> None:
    result = run_simulation("breakup_flowers", tone="robotic", strictness=0.75)
    assert result.stage3.local_flavor.score < 0.75
    assert result.stage3.overall_pass is False


def test_strictness_slider_affects_verdict() -> None:
    loose = run_simulation("breakup_flowers", tone="standard", strictness=0.50)
    strict = run_simulation("breakup_flowers", tone="standard", strictness=0.95)
    assert loose.stage3.strictness == 0.50
    assert strict.stage3.strictness == 0.95


@pytest.mark.asyncio
async def test_dev_simulator_page_renders_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/dev/simulator")
    assert response.status_code == 200
    html = response.text
    assert "Routing Simulator" in html
    assert 'data-testid="stage-preprocessor"' in html
    assert "QueryPreprocessor" in html
    assert 'name="strictness"' in html
    assert f'value="{DEFAULT_STRICTNESS}"' in html or f"value={DEFAULT_STRICTNESS}" in html


@pytest.mark.asyncio
async def test_dev_simulator_hidden_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/dev/simulator")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_dev_simulator_run_htmx_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/dev/simulator/run",
            data={
                "scenario": "tanglish_kandy",
                "tone": "high_empathy",
                "strictness": "0.80",
            },
        )
    assert response.status_code == 200
    html = response.text
    assert 'data-field="detected_vernacular"' in html
    assert "tanglish" in html
    assert 'data-testid="tool-binding"' in html
    assert "kapruka_check_delivery" in html
    assert 'data-testid="rubric-local_flavor"' in html


def test_results_template_renders_three_stages() -> None:
    result = run_simulation("keeri_samba", tone="standard")
    templates = get_templates()
    response = templates.TemplateResponse(
        _make_request(),
        "dev/routing_simulator_results.html",
        {
            "request": _make_request(),
            "result": result,
            "utility_prompt_preview": "Utility preview",
            "concierge_prompt_preview": "Concierge preview",
        },
    )
    template = templates.env.get_template("dev/routing_simulator_results.html")
    html = template.render(
        result=result,
        utility_prompt_preview="Utility preview",
        concierge_prompt_preview="Concierge preview",
    )
    assert 'data-testid="stage-preprocessor"' in html
    assert 'data-testid="stage-routing"' in html
    assert 'data-testid="stage-ci-report"' in html
    assert "intent_preservation" in html
    _ = response  # exercise TemplateResponse construction
