"""LangGraph preprocessor routing simulator for the dev dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evals.llm_judge import (
    RubricScore,
    score_constraint_fidelity,
    score_intent_preservation,
    score_local_flavor,
)

from lib.chat.intent_metadata import IntentMetadata
from lib.chat.query_preprocessor import QueryPreprocessor, classify_query_mode
from lib.chat.system_prompts import (
    LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION,
    UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION,
    select_response_system_instruction,
)
from lib.neo4j.hybrid_context import discovery_tool_manifest

ScenarioId = Literal["keeri_samba", "tanglish_kandy", "breakup_flowers"]
ToneProfileId = Literal["robotic", "standard", "high_empathy"]

DEFAULT_STRICTNESS = 0.75
MIN_STRICTNESS = 0.50
MAX_STRICTNESS = 0.95


@dataclass(frozen=True, slots=True)
class SimulatorScenario:
    """Preset query fixture for the routing simulator."""

    id: ScenarioId
    label: str
    query: str
    preserved_terms: tuple[str, ...]
    must_contain: tuple[str, ...]
    query_mode: Literal["utility", "situational"]


SCENARIOS: dict[ScenarioId, SimulatorScenario] = {
    "keeri_samba": SimulatorScenario(
        id="keeri_samba",
        label="Keeri Samba utility query",
        query="show me Keeri Samba rice price in Colombo",
        preserved_terms=("keeri", "samba"),
        must_contain=(),
        query_mode="utility",
    ),
    "tanglish_kandy": SimulatorScenario(
        id="tanglish_kandy",
        label="Tanglish Kandy delivery query",
        query="mage girlfriend ku birthday cake ona, deliver to Kandy puluvan da?",
        preserved_terms=("cake", "Kandy"),
        must_contain=("Kandy",),
        query_mode="utility",
    ),
    "breakup_flowers": SimulatorScenario(
        id="breakup_flowers",
        label="Highly situational breakup flowers",
        query="broke up with my girlfriend yesterday, need flowers to apologize",
        preserved_terms=("flower",),
        must_contain=(),
        query_mode="situational",
    ),
}

TONE_PROFILES: dict[ToneProfileId, str] = {
    "robotic": "Robotic / Formal",
    "standard": "Standard E-com Help",
    "high_empathy": "High Empathy + Sri Lankan Flavor",
}

_TONE_RESPONSE_BODIES: dict[ToneProfileId, str] = {
    "robotic": (
        "Dear valued customer, please be advised that we have catalog options "
        "matching your request."
    ),
    "standard": ("Here are matching gift options with prices and stock from our Kapruka catalog."),
    "high_empathy": (
        "Aiyo machan, hodata gentle choice — here are thoughtful options for this moment."
    ),
}

_preprocessor = QueryPreprocessor()


@dataclass(frozen=True, slots=True)
class SimulatorStage2:
    """Prompt routing and discovery tool binding."""

    prompt_template: str
    prompt_excerpt: str
    tools: tuple[str, ...]
    tool_binding_label: str


@dataclass(frozen=True, slots=True)
class SimulatorStage3:
    """CI tone-gate report card against adjustable strictness."""

    mock_response_html: str
    intent_preservation: RubricScore
    constraint_fidelity: RubricScore
    local_flavor: RubricScore
    strictness: float
    overall_pass: bool


@dataclass(frozen=True, slots=True)
class SimulatorResult:
    """Full three-stage simulator output."""

    scenario: SimulatorScenario
    tone_profile: ToneProfileId
    tone_label: str
    intent_metadata: IntentMetadata
    stage2: SimulatorStage2
    stage3: SimulatorStage3


def _clamp_strictness(value: float) -> float:
    return max(MIN_STRICTNESS, min(MAX_STRICTNESS, value))


def _prompt_excerpt(intent_metadata: IntentMetadata) -> str:
    instruction = select_response_system_instruction(intent_metadata)
    marker = "gift concierge" if intent_metadata.get("is_situational") else "fast, transactional"
    first_line = next(
        (line.strip() for line in instruction.splitlines() if line.strip()),
        instruction[:120],
    )
    return f"{marker} — {first_line[:100]}"


def _tool_binding_label(tools: frozenset[str]) -> str:
    if "kapruka_check_delivery" in tools:
        return "search + kapruka_check_delivery"
    return "search-only"


def build_mock_response_html(
    tone: ToneProfileId,
    scenario: SimulatorScenario,
    intent_metadata: IntentMetadata,
) -> str:
    """Synthetic assistant HTML for CI rubric scoring (no live LLM)."""
    body = _TONE_RESPONSE_BODIES[tone]
    if intent_metadata.get("target_city"):
        body += f" Delivery to {intent_metadata['target_city']} is available."
    if scenario.preserved_terms:
        body += " " + " ".join(scenario.preserved_terms)
    return (
        '<div class="flex justify-start" aria-label="Assistant message" '
        'data-role="assistant-message">'
        f"<p>{body}</p></div>"
    )


def run_simulation(
    scenario_id: ScenarioId,
    *,
    tone: ToneProfileId = "standard",
    strictness: float = DEFAULT_STRICTNESS,
) -> SimulatorResult:
    """Run preprocessor → routing → CI report for one scenario."""
    scenario = SCENARIOS[scenario_id]
    threshold = _clamp_strictness(strictness)

    intent_metadata = _preprocessor.process(scenario.query)
    tools = discovery_tool_manifest(intent_metadata)
    tool_list = sorted(tools)

    prompt_template = (
        "Localized Concierge" if intent_metadata.get("is_situational") else "Utility E-commerce"
    )
    stage2 = SimulatorStage2(
        prompt_template=prompt_template,
        prompt_excerpt=_prompt_excerpt(intent_metadata),
        tools=tuple(tool_list),
        tool_binding_label=_tool_binding_label(tools),
    )

    mock_html = build_mock_response_html(tone, scenario, intent_metadata)
    query_mode: Literal["utility", "situational"] = (
        "situational" if intent_metadata.get("is_situational") else "utility"
    )

    intent_score = score_intent_preservation(
        mock_html,
        list(scenario.preserved_terms),
        threshold=threshold,
    )
    constraint_score = score_constraint_fidelity(
        mock_html,
        must_contain=list(scenario.must_contain),
        threshold=threshold,
    )
    flavor_score = score_local_flavor(
        mock_html,
        query_mode=query_mode,
        threshold=threshold,
    )

    overall_pass = (
        intent_score.verdict == "pass"
        and constraint_score.verdict == "pass"
        and flavor_score.verdict == "pass"
    )

    stage3 = SimulatorStage3(
        mock_response_html=mock_html,
        intent_preservation=intent_score,
        constraint_fidelity=constraint_score,
        local_flavor=flavor_score,
        strictness=threshold,
        overall_pass=overall_pass,
    )

    return SimulatorResult(
        scenario=scenario,
        tone_profile=tone,
        tone_label=TONE_PROFILES[tone],
        intent_metadata=intent_metadata,
        stage2=stage2,
        stage3=stage3,
    )


def utility_prompt_preview() -> str:
    """First line of the Utility E-commerce template for legend display."""
    return UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION.strip().splitlines()[0]


def concierge_prompt_preview() -> str:
    """First line of the Localized Concierge template for legend display."""
    return LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION.strip().splitlines()[0]


def scenario_query_mode(query: str) -> Literal["utility", "situational"]:
    """Expose classify_query_mode for tests."""
    return classify_query_mode(query)
