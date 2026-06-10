"""Rubric-based LLM-as-judge helpers for HybridRAG E2E and shadow tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def text(self) -> str:
        return " ".join(self._parts)


def _html_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html or "")
    return parser.text()


def _has_testid(html: str, testid: str) -> bool:
    return f'data-testid="{testid}"' in html


JudgeVerdict = Literal["pass", "fail"]

_LOCAL_FLAVOR_MARKERS: tuple[str, ...] = (
    "aiyo",
    "machan",
    "malli",
    "nangi",
    "hodata",
    "kiyala",
    "ammata",
    "mage",
    "mama",
)
_FORMAL_ONLY_MARKERS: tuple[str, ...] = (
    "dear valued customer",
    "we regret to inform",
    "please be advised",
    "kindly note that",
)


@dataclass(frozen=True, slots=True)
class RubricScore:
    """Single rubric dimension scored 0.0–1.0."""

    name: str
    score: float
    verdict: JudgeVerdict
    reason: str


@dataclass(frozen=True, slots=True)
class FidelityAssessment:
    """DOM + MCP alignment for one chat turn."""

    tool_alignment: RubricScore
    visual_fidelity: RubricScore
    constraint_fidelity: RubricScore | None = None


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _verdict(score: float, threshold: float) -> JudgeVerdict:
    return "pass" if score >= threshold else "fail"


def score_mcp_tool_alignment(
    called_tools: list[str],
    expected_tools: list[str],
    *,
    threshold: float = 1.0,
) -> RubricScore:
    """Verify the agent invoked the expected Kapruka MCP tools."""
    expected = set(expected_tools)
    called = set(called_tools)
    if not expected:
        return RubricScore(
            name="mcp_tool_alignment",
            score=1.0,
            verdict="pass",
            reason="No MCP tools required for this turn.",
        )

    matched = expected & called
    score = _clamp(len(matched) / len(expected))
    missing = sorted(expected - called)
    reason = (
        f"Matched {len(matched)}/{len(expected)} expected tools."
        if not missing
        else f"Missing tools: {', '.join(missing)}; called: {', '.join(called) or 'none'}."
    )
    return RubricScore(
        name="mcp_tool_alignment",
        score=score,
        verdict=_verdict(score, threshold),
        reason=reason,
    )


def score_visual_fidelity(
    response_html: str,
    *,
    require_product_card: bool = False,
    require_carousel: bool = False,
    require_payment_cta: bool = False,
    threshold: float = 0.7,
) -> RubricScore:
    """Score structured HTMX partials instead of plain-text walls."""
    html = response_html or ""
    checks: list[tuple[str, bool, float]] = []

    has_assistant = 'aria-label="Assistant message"' in html
    checks.append(("assistant bubble", has_assistant, 0.25))

    if require_product_card:
        has_card = _has_testid(html, "product-card")
        checks.append(("product card", has_card, 0.35))
    if require_carousel:
        has_carousel = _has_testid(html, "product-carousel")
        checks.append(("product carousel", has_carousel, 0.2))
    if require_payment_cta:
        has_cta = _has_testid(html, "checkout-payment-cta")
        checks.append(("payment CTA", has_cta, 0.35))

    has_image = bool(re.search(r"<img[^>]+src=", html, re.I))
    if require_product_card and not require_payment_cta:
        checks.append(("product image", has_image, 0.2))

    if not checks:
        plain_len = len(_html_text(html))
        score = 1.0 if plain_len < 800 else 0.5
        return RubricScore(
            name="visual_fidelity",
            score=score,
            verdict=_verdict(score, threshold),
            reason="Text-only response within acceptable length."
            if score >= threshold
            else "Response is a long text wall without structured UI.",
        )

    total_weight = sum(weight for _, _, weight in checks)
    earned = sum(weight for _, ok, weight in checks if ok)
    score = _clamp(earned / total_weight if total_weight else 0.0)
    failed = [name for name, ok, _ in checks if not ok]
    reason = (
        "Structured HTMX partials present."
        if not failed
        else f"Missing UI elements: {', '.join(failed)}."
    )
    return RubricScore(
        name="visual_fidelity",
        score=score,
        verdict=_verdict(score, threshold),
        reason=reason,
    )


def score_constraint_fidelity(
    response_html: str,
    *,
    must_contain: list[str] | None = None,
    must_not_contain: list[str] | None = None,
    threshold: float = 1.0,
) -> RubricScore:
    """Check rendered HTML honors user constraints (city, dietary, etc.)."""
    text = _html_text(response_html).lower()
    must_contain = must_contain or []
    must_not_contain = must_not_contain or []

    missing = [term for term in must_contain if term.lower() not in text]
    forbidden = [term for term in must_not_contain if term.lower() in text]

    penalty = len(missing) + len(forbidden)
    score = _clamp(1.0 - (0.5 * penalty))
    parts: list[str] = []
    if missing:
        parts.append(f"missing terms: {', '.join(missing)}")
    if forbidden:
        parts.append(f"forbidden terms present: {', '.join(forbidden)}")
    reason = "; ".join(parts) if parts else "All user constraints reflected in DOM text."
    return RubricScore(
        name="constraint_fidelity",
        score=score,
        verdict=_verdict(score, threshold),
        reason=reason,
    )


def score_local_flavor(
    response_text: str,
    *,
    query_mode: Literal["utility", "situational"] = "utility",
    threshold: float = 0.7,
) -> RubricScore:
    """Tone rubric: situational queries should feel local, not corporate."""
    lowered = response_text.lower()
    if query_mode == "utility":
        return RubricScore(
            name="local_flavor",
            score=1.0,
            verdict="pass",
            reason="Utility query — formal tone acceptable.",
        )

    local_hits = sum(1 for marker in _LOCAL_FLAVOR_MARKERS if marker in lowered)
    formal_hits = sum(1 for marker in _FORMAL_ONLY_MARKERS if marker in lowered)
    sinhala_chars = len(re.findall(r"[\u0D80-\u0DFF]", response_text))

    score = _clamp(0.2 + local_hits * 0.2 + (0.3 if sinhala_chars else 0.0) - formal_hits * 0.3)
    reason = f"local markers={local_hits}, sinhala_chars={sinhala_chars}, formal={formal_hits}."
    return RubricScore(
        name="local_flavor",
        score=score,
        verdict=_verdict(score, threshold),
        reason=reason,
    )


def score_intent_preservation(
    final_html: str,
    preserved_terms: list[str],
    *,
    threshold: float = 0.8,
) -> RubricScore:
    """Shadow-test: initial user goal still visible in the final DOM/cart state."""
    text = _html_text(final_html).lower()
    if not preserved_terms:
        return RubricScore(
            name="intent_preservation",
            score=1.0,
            verdict="pass",
            reason="No preservation terms required.",
        )
    matched = [term for term in preserved_terms if term.lower() in text]
    score = _clamp(len(matched) / len(preserved_terms))
    missing = [term for term in preserved_terms if term.lower() not in text]
    reason = (
        f"Preserved {len(matched)}/{len(preserved_terms)} intent terms."
        if not missing
        else f"Intent drift — missing: {', '.join(missing)}."
    )
    return RubricScore(
        name="intent_preservation",
        score=score,
        verdict=_verdict(score, threshold),
        reason=reason,
    )
