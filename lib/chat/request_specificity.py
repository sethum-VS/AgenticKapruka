"""Hybrid request-specificity scorer for pre-search discovery gating."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal, Mapping

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from lib.chat.delivery_dates import is_ambiguous_weekday_phrase
from lib.chat.intent_heuristics import (
    _GIFT_SPECIFIC_RE,
    _VAGUE_GIFT_RE,
    PROCEED_CHECKOUT_MESSAGE,
    Intent,
    classify_routing_guard,
    is_bare_category_pivot,
    is_budgeted_gift_ideas_message,
    is_guest_checkout_question,
)
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.model_router import FLASH_MODEL
from lib.chat.off_topic import is_impossible_catalog_request, is_off_topic_message
from lib.chat.support_faq import is_support_question
from lib.genai.fallback import generate_content_with_fallback
from lib.kapruka.product_id import contains_product_id
from lib.neo4j.hybrid_context import extract_budget, extract_max_price

logger = logging.getLogger(__name__)

SpecificityBand = Literal["proceed", "clarify", "ambiguous"]
ClarificationDimension = Literal["product", "occasion", "budget"]

# Calibrated in tests/unit/test_request_specificity.py
PROCEED_THRESHOLD = 45.0
CLARIFY_THRESHOLD = 25.0

_WEIGHT_PRODUCT = 0.40
_WEIGHT_OCCASION = 0.35
_WEIGHT_BUDGET = 0.25

_DIMENSION_PRIORITY: tuple[ClarificationDimension, ...] = ("product", "occasion", "budget")

_PRODUCT_CATEGORY_RE = re.compile(
    r"\b(?:cake|cupcakes?|flower|flowers|rose|roses|bouquet|chocolate|chocolates|"
    r"hamper|voucher|combo|combopack)\b",
    re.I,
)
_VAGUE_GIFT_NOUN_RE = re.compile(r"\b(?:gift|gifts|present|presents)\b", re.I)
_OCCASION_RE = re.compile(
    r"\b(?:birthday|anniversary|wedding|valentine|graduation|new baby|baby shower|retirement)\b",
    re.I,
)
_RECIPIENT_RE = re.compile(
    r"\b(?:wife|husband|mom|mother|mum|dad|father|girlfriend|boyfriend|partner|"
    r"sister|brother|son|daughter|grandma|grandmother|grandpa|grandfather|colleague|"
    r"her|him|them)\b",
    re.I,
)
_FOR_RECIPIENT_RE = re.compile(r"\bfor\s+(?:her|him|them)\b", re.I)

_PRODUCT_QUESTION = (
    "What type of gift — flowers, cake, voucher, or hamper? "
    "For example: 'birthday cake for mom under Rs 5,000'."
)
_OCCASION_QUESTION = (
    "Who is it for, and is there an occasion (birthday, anniversary)?"
)
_BUDGET_QUESTION = "Do you have a budget in mind (e.g. under Rs 5,000)?"
_EMPTY_MESSAGE_QUESTION = "What are you looking for today?"


@dataclass(frozen=True)
class SpecificityResult:
    """Heuristic or LLM-refined specificity assessment for a discovery turn."""

    score: float
    dimension_scores: dict[str, float]
    missing_dimension: ClarificationDimension | None
    band: SpecificityBand
    clarifying_question: str | None


class SpecificityRefinement(BaseModel):
    """Structured Gemini response for ambiguous-band refinement."""

    score: float = Field(ge=0.0, le=100.0)
    product_score: float = Field(ge=0.0, le=1.0)
    occasion_score: float = Field(ge=0.0, le=1.0)
    budget_score: float = Field(ge=0.0, le=1.0)
    missing_dimension: ClarificationDimension | None = None
    band: Literal["proceed", "clarify"]


_REFINEMENT_SYSTEM = """\
You refine how actionable a Kapruka gift-shopping request is before catalog search.

Score three dimensions from 0.0 to 1.0:
- product: specific product type (cake, flowers, chocolate, hamper, voucher, product id)
- occasion: recipient and/or occasion (birthday, anniversary, mom, dad, colleague)
- budget: explicit price cap in the message or session

Return adjusted score (weighted: product 40%, occasion 35%, budget 25%) and band:
- proceed when the shopper gave enough to search without a bloated carousel
- clarify when one focused follow-up is still needed

Pick missing_dimension as the weakest dimension below 1.0 (priority: product, occasion, budget).
Do not ask about delivery city or date.
"""


def should_bypass_specificity_scorer(
    message: str,
    *,
    guard_intent: Intent | None = None,
) -> bool:
    """True when an existing fast path should skip pre-search specificity scoring."""
    stripped = message.strip()
    if not stripped:
        return True
    if guard_intent is not None:
        return True
    if is_guest_checkout_question(stripped):
        return True
    if classify_routing_guard(stripped) is not None:
        return True
    if stripped == PROCEED_CHECKOUT_MESSAGE:
        return True
    if is_support_question(stripped):
        return True
    if is_off_topic_message(stripped) or is_impossible_catalog_request(stripped):
        return True
    if contains_product_id(stripped):
        return True
    if is_ambiguous_weekday_phrase(stripped):
        return True
    return False


_GENERIC_WANTS_SOMETHING_RE = re.compile(
    r"\bsomething\s+(?:nice|good|great|special|beautiful|pretty|awesome|lovely|cool|thoughtful|unique)\b"
    r"|\banything\s+(?:nice|good|great|special|beautiful|pretty|awesome|lovely|cool|thoughtful)\b"
    r"|\bi\s+(?:want|need)\s+to\s+(?:buy|get|order|send)\s+something\b",
    re.I,
)
_BARE_GIFT_NEED_RE = re.compile(
    r"\bi\s+(?:need|want)\s+(?:a\s+)?(?:gift|present)\b",
    re.I,
)


def _is_extra_vague_gift_query(message: str) -> bool:
    stripped = message.strip()
    return bool(
        _VAGUE_GIFT_RE.search(stripped)
        or _GENERIC_WANTS_SOMETHING_RE.search(stripped)
        or _BARE_GIFT_NEED_RE.search(stripped)
    )


def _score_product_dimension(
    message: str,
    *,
    session_product_focus: str | None,
    session_flavor_hint: str | None,
    session_recipient_hint: str | None,
    intent_metadata: IntentMetadata,
) -> float:
    stripped = message.strip()
    if not stripped:
        return 0.0
    if contains_product_id(stripped):
        return 1.0
    if _is_extra_vague_gift_query(stripped):
        return 0.0
    if _PRODUCT_CATEGORY_RE.search(stripped) or _GIFT_SPECIFIC_RE.search(stripped):
        return 1.0
    if is_bare_category_pivot(stripped) is not None:
        return 1.0
    if not _is_extra_vague_gift_query(stripped):
        focus = (session_product_focus or "").strip().lower()
        if focus and focus != "gift":
            return 1.0
    flavor = (session_flavor_hint or intent_metadata.get("session_flavor_hint") or "").strip()
    if flavor:
        return 0.5
    if _VAGUE_GIFT_NOUN_RE.search(stripped):
        if _RECIPIENT_RE.search(stripped) or session_recipient_hint:
            return 1.0
        return 0.5
    return 0.0


def _score_occasion_dimension(
    message: str,
    *,
    session_occasion: str | None,
    session_recipient_hint: str | None,
) -> float:
    stripped = message.strip()
    has_occasion = bool(_OCCASION_RE.search(stripped))
    has_recipient = bool(_RECIPIENT_RE.search(stripped) or _FOR_RECIPIENT_RE.search(stripped))
    # Only inherit stale session context for follow-up / pivot turns, not generic fresh queries.
    # Inheriting context for "I want to buy something nice" inflates score and skips clarification.
    if not _is_extra_vague_gift_query(stripped):
        has_occasion = has_occasion or bool(session_occasion)
        has_recipient = has_recipient or bool(session_recipient_hint)
    if has_occasion and has_recipient:
        return 1.0
    if has_recipient:
        return 1.0
    if has_occasion:
        return 0.5
    return 0.0


def _score_delivery_dimension(
    message: str,
    *,
    intent_metadata: IntentMetadata,
    session_delivery_date: str | None = None,
) -> float:
    """Soft signal: city and/or date known from message or session."""
    from lib.chat.delivery_dates import normalize_delivery_date
    from lib.chat.query_preprocessor import extract_target_city

    stripped = message.strip()
    has_city = bool(intent_metadata.get("target_city")) or extract_target_city(stripped) is not None
    has_date = (
        normalize_delivery_date({}, stripped) is not None
        or bool(
            isinstance(session_delivery_date, str) and session_delivery_date.strip(),
        )
    )
    if has_city and has_date:
        return 1.0
    if has_city or has_date:
        return 0.5
    return 0.0


def is_delivery_only_inquiry(
    message: str,
    *,
    intent_metadata: IntentMetadata | None = None,
) -> bool:
    """True when the turn is only about delivery area/date/fees, not product discovery."""
    from lib.chat.query_preprocessor import (
        QueryPreprocessor,
        _has_perishable_gift_intent,
    )

    meta: IntentMetadata = intent_metadata or QueryPreprocessor().process(message)
    if not meta.get("requires_delivery_validation"):
        return False
    stripped = message.strip()
    if _has_perishable_gift_intent(stripped) or contains_product_id(stripped):
        return False
    delivery_score = _score_delivery_dimension(
        stripped,
        intent_metadata=meta,
        session_delivery_date=meta.get("session_delivery_date") or meta.get("delivery_date"),
    )
    if delivery_score < 1.0:
        return False
    product_score = _score_product_dimension(
        stripped,
        session_product_focus=None,
        session_flavor_hint=None,
        session_recipient_hint=None,
        intent_metadata=meta,
    )
    occasion_score = _score_occasion_dimension(
        stripped,
        session_occasion=None,
        session_recipient_hint=None,
    )
    return product_score < 0.5 and occasion_score < 0.5


def _score_budget_dimension(
    message: str,
    *,
    session_budget_max: float | None,
    intent_metadata: IntentMetadata,
) -> float:
    stripped = message.strip()
    if extract_budget(stripped) is not None:
        return 1.0
    if is_budgeted_gift_ideas_message(stripped):
        return 1.0
    budget_meta = intent_metadata.get("budget_max")
    if isinstance(budget_meta, (int, float)) and budget_meta > 0:
        if not _is_extra_vague_gift_query(stripped):
            return 1.0
    if isinstance(session_budget_max, (int, float)) and session_budget_max > 0:
        if not _is_extra_vague_gift_query(stripped):
            return 1.0
    if extract_max_price(stripped) is not None:
        return 0.5
    return 0.0


def _weighted_score(dimension_scores: dict[str, float]) -> float:
    return (
        dimension_scores.get("product", 0.0) * _WEIGHT_PRODUCT
        + dimension_scores.get("occasion", 0.0) * _WEIGHT_OCCASION
        + dimension_scores.get("budget", 0.0) * _WEIGHT_BUDGET
    ) * 100.0


def _resolve_band(
    score: float,
    dimension_scores: dict[str, float],
    *,
    message: str = "",
) -> SpecificityBand:
    budget = dimension_scores.get("budget", 0.0)
    product = dimension_scores.get("product", 0.0)
    if budget >= 1.0 and product >= 0.5:
        return "proceed"
    if score >= PROCEED_THRESHOLD:
        return "proceed"
    if score < CLARIFY_THRESHOLD:
        return "clarify"
    return "ambiguous"


def _pick_missing_dimension(dimension_scores: dict[str, float]) -> ClarificationDimension | None:
    incomplete = {
        dim: dimension_scores.get(dim, 0.0)
        for dim in _DIMENSION_PRIORITY
        if dimension_scores.get(dim, 0.0) < 1.0
    }
    if not incomplete:
        return None
    min_score = min(incomplete.values())
    for dim in _DIMENSION_PRIORITY:
        if incomplete.get(dim) == min_score:
            return dim
    return None


def build_clarifying_question(
    dimension: ClarificationDimension | None,
    *,
    is_situational: bool = False,
) -> str:
    """Public wrapper for dimension-specific clarifying copy."""
    return _build_clarifying_question(dimension, is_situational=is_situational)


def _build_clarifying_question(
    dimension: ClarificationDimension | None,
    *,
    is_situational: bool = False,
) -> str:
    if dimension == "occasion":
        question = _OCCASION_QUESTION
    elif dimension == "budget":
        question = _BUDGET_QUESTION
    elif dimension == "product":
        question = _PRODUCT_QUESTION
    else:
        question = _EMPTY_MESSAGE_QUESTION
    if is_situational:
        return f"I'm sorry to hear you're going through this. {question}"
    return question


def score_request_specificity(
    message: str,
    *,
    session_product_focus: str | None,
    session_occasion: str | None,
    session_recipient_hint: str | None,
    session_budget_max: float | None,
    session_flavor_hint: str | None = None,
    intent_metadata: IntentMetadata | None = None,
) -> SpecificityResult:
    """Heuristic specificity score from the current message plus session slots."""
    meta: IntentMetadata = intent_metadata or {}
    stripped = message.strip()
    if not stripped:
        dimension_scores = {"product": 0.0, "occasion": 0.0, "budget": 0.0}
        return SpecificityResult(
            score=0.0,
            dimension_scores=dimension_scores,
            missing_dimension="product",
            band="clarify",
            clarifying_question=_EMPTY_MESSAGE_QUESTION,
        )

    dimension_scores = {
        "product": _score_product_dimension(
            stripped,
            session_product_focus=session_product_focus,
            session_flavor_hint=session_flavor_hint,
            session_recipient_hint=session_recipient_hint,
            intent_metadata=meta,
        ),
        "occasion": _score_occasion_dimension(
            stripped,
            session_occasion=session_occasion,
            session_recipient_hint=session_recipient_hint,
        ),
        "budget": _score_budget_dimension(
            stripped,
            session_budget_max=session_budget_max,
            intent_metadata=meta,
        ),
    }
    score = _weighted_score(dimension_scores)
    delivery_score = _score_delivery_dimension(
        stripped,
        intent_metadata=meta,
        session_delivery_date=meta.get("session_delivery_date") or meta.get("delivery_date"),
    )
    if meta.get("target_city") and (
        meta.get("session_delivery_date")
        or meta.get("delivery_date")
    ):
        score = min(100.0, score + 5.0)
    band = _resolve_band(score, dimension_scores, message=stripped)
    if (
        dimension_scores.get("product", 0.0) >= 1.0
        and dimension_scores.get("occasion", 0.0) >= 0.5
        and (
            dimension_scores.get("budget", 0.0) >= 0.5
            or bool(meta.get("session_delivery_date") or meta.get("delivery_date"))
        )
    ):
        band = "proceed"
    if (
        dimension_scores.get("product", 0.0) >= 0.5
        and dimension_scores.get("occasion", 0.0) >= 0.5
    ):
        band = "proceed"
    if (
        dimension_scores.get("product", 0.0) >= 1.0
        and dimension_scores.get("occasion", 0.0) < 0.5
        and dimension_scores.get("budget", 0.0) < 1.0
        and delivery_score < 1.0
    ):
        band = "clarify"
    if dimension_scores.get("product", 0.0) >= 1.0 and delivery_score >= 1.0:
        band = "proceed"
    if is_delivery_only_inquiry(stripped, intent_metadata=meta):
        band = "proceed"
    if (
        dimension_scores.get("budget", 0.0) >= 1.0
        and dimension_scores.get("product", 0.0) < 0.5
        and dimension_scores.get("occasion", 0.0) < 0.5
    ):
        band = "clarify"
    missing = _pick_missing_dimension(dimension_scores)
    question = (
        _build_clarifying_question(missing, is_situational=bool(meta.get("is_situational")))
        if band in ("clarify", "ambiguous")
        else None
    )
    return SpecificityResult(
        score=score,
        dimension_scores=dimension_scores,
        missing_dimension=missing,
        band=band,
        clarifying_question=question,
    )


def _session_summary(
    *,
    session_product_focus: str | None,
    session_occasion: str | None,
    session_recipient_hint: str | None,
    session_budget_max: float | None,
) -> str:
    parts: list[str] = []
    if session_product_focus:
        parts.append(f"product_focus={session_product_focus}")
    if session_occasion:
        parts.append(f"occasion={session_occasion}")
    if session_recipient_hint:
        parts.append(f"recipient={session_recipient_hint}")
    if isinstance(session_budget_max, (int, float)) and session_budget_max > 0:
        parts.append(f"budget_max={session_budget_max}")
    return ", ".join(parts) if parts else "none"


def _result_from_refinement(
    refinement: SpecificityRefinement,
    *,
    is_situational: bool,
) -> SpecificityResult:
    dimension_scores = {
        "product": refinement.product_score,
        "occasion": refinement.occasion_score,
        "budget": refinement.budget_score,
    }
    band: SpecificityBand = refinement.band
    missing = refinement.missing_dimension or _pick_missing_dimension(dimension_scores)
    question = (
        _build_clarifying_question(missing, is_situational=is_situational)
        if band == "clarify"
        else None
    )
    return SpecificityResult(
        score=refinement.score,
        dimension_scores=dimension_scores,
        missing_dimension=missing,
        band=band,
        clarifying_question=question,
    )


async def refine_specificity_with_llm(
    message: str,
    heuristic: SpecificityResult,
    *,
    genai_client: object,
    session_product_focus: str | None = None,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
    session_budget_max: float | None = None,
    intent_metadata: IntentMetadata | None = None,
) -> SpecificityResult:
    """Gemini flash refinement for ambiguous heuristic band; safe clarify on failure."""
    meta: IntentMetadata = intent_metadata or {}
    if not isinstance(genai_client, genai.Client):
        return _fallback_clarify(heuristic, is_situational=bool(meta.get("is_situational")))

    session_line = _session_summary(
        session_product_focus=session_product_focus,
        session_occasion=session_occasion,
        session_recipient_hint=session_recipient_hint,
        session_budget_max=session_budget_max,
    )
    user_prompt = (
        f"Message: {message.strip()!r}\n"
        f"Session context: {session_line}\n"
        f"Heuristic score: {heuristic.score:.1f}\n"
        f"Heuristic dimensions: {heuristic.dimension_scores}\n"
        f"Heuristic band: {heuristic.band}"
    )

    try:
        response = generate_content_with_fallback(
            client=genai_client,
            model=FLASH_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_REFINEMENT_SYSTEM,
                response_mime_type="application/json",
                response_schema=SpecificityRefinement,
                temperature=0,
            ),
        )
    except Exception:
        logger.warning("refine_specificity_with_llm: Gemini call failed", exc_info=True)
        return _fallback_clarify(heuristic, is_situational=bool(meta.get("is_situational")))

    parsed: SpecificityRefinement | None = None
    if response.parsed is not None:
        try:
            if isinstance(response.parsed, SpecificityRefinement):
                parsed = response.parsed
            else:
                parsed = SpecificityRefinement.model_validate(response.parsed)
        except ValidationError:
            parsed = None

    if parsed is None:
        raw_text = (response.text or "").strip()
        if raw_text:
            try:
                parsed = SpecificityRefinement.model_validate_json(raw_text)
            except Exception:
                logger.debug("refine_specificity_with_llm: invalid JSON %r", raw_text)
        if parsed is None:
            return _fallback_clarify(heuristic, is_situational=bool(meta.get("is_situational")))

    return _result_from_refinement(parsed, is_situational=bool(meta.get("is_situational")))


def _fallback_clarify(
    heuristic: SpecificityResult,
    *,
    is_situational: bool,
) -> SpecificityResult:
    missing = heuristic.missing_dimension or _pick_missing_dimension(heuristic.dimension_scores)
    return SpecificityResult(
        score=heuristic.score,
        dimension_scores=heuristic.dimension_scores,
        missing_dimension=missing,
        band="clarify",
        clarifying_question=_build_clarifying_question(missing, is_situational=is_situational),
    )


def resolve_awaiting_clarification_dimension(
    state: Mapping[str, object],
) -> ClarificationDimension | None:
    """Map legacy gift-preferences flag to the new awaiting-dimension slot."""
    awaiting = state.get("session_awaiting_clarification_dimension")
    if awaiting in ("product", "occasion", "budget"):
        return awaiting
    if state.get("session_awaiting_gift_preferences"):
        return "product"
    return None
