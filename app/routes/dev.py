"""Development-only routes (simulator dashboard)."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from starlette.responses import Response

from app.templating import get_templates
from lib.dev.routing_simulator import (
    DEFAULT_STRICTNESS,
    MAX_STRICTNESS,
    MIN_STRICTNESS,
    SCENARIOS,
    TONE_PROFILES,
    ScenarioId,
    SimulatorResult,
    ToneProfileId,
    concierge_prompt_preview,
    run_simulation,
    utility_prompt_preview,
)

router = APIRouter()


def _is_production() -> bool:
    return os.getenv("APP_ENV", "development").lower() == "production"


async def require_dev_environment() -> None:
    """Hide dev tooling in production deployments."""
    if _is_production():
        raise HTTPException(status_code=404, detail="Not found")


DevOnly = Annotated[None, Depends(require_dev_environment)]


def _parse_strictness(raw: float | str | None) -> float:
    if raw is None:
        return DEFAULT_STRICTNESS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_STRICTNESS
    return max(MIN_STRICTNESS, min(MAX_STRICTNESS, value))


def _simulator_context(
    request: Request,
    result: SimulatorResult,
) -> dict[str, object]:
    return {
        "request": request,
        "result": result,
        "scenarios": SCENARIOS,
        "tone_profiles": TONE_PROFILES,
        "utility_prompt_preview": utility_prompt_preview(),
        "concierge_prompt_preview": concierge_prompt_preview(),
    }


@router.get("/simulator", dependencies=[Depends(require_dev_environment)])
async def routing_simulator_page(
    request: Request,
    _dev: DevOnly = None,
) -> Response:
    """Interactive LangGraph preprocessor routing simulator."""
    templates = get_templates()
    default = run_simulation("keeri_samba")
    return templates.TemplateResponse(
        request,
        "dev/routing_simulator.html",
        {
            **_simulator_context(request, default),
            "selected_scenario": "keeri_samba",
            "selected_tone": "standard",
            "strictness": DEFAULT_STRICTNESS,
            "min_strictness": MIN_STRICTNESS,
            "max_strictness": MAX_STRICTNESS,
        },
    )


@router.post("/simulator/run", dependencies=[Depends(require_dev_environment)])
async def routing_simulator_run(
    request: Request,
    scenario: Annotated[ScenarioId, Form()] = "keeri_samba",
    tone: Annotated[ToneProfileId, Form()] = "standard",
    strictness: Annotated[float, Form()] = DEFAULT_STRICTNESS,
    _dev: DevOnly = None,
) -> Response:
    """HTMX partial: rerun preprocessor routing + CI report card."""
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=422, detail="Unknown scenario")
    if tone not in TONE_PROFILES:
        raise HTTPException(status_code=422, detail="Unknown tone profile")

    templates = get_templates()
    result = run_simulation(
        scenario,
        tone=tone,
        strictness=_parse_strictness(strictness),
    )
    return templates.TemplateResponse(
        request,
        "dev/routing_simulator_results.html",
        _simulator_context(request, result),
    )
