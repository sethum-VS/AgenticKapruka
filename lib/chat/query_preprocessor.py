"""Pre-graph query signals: utility vs situational mode and vernacular hints."""

from __future__ import annotations

import re
from typing import Literal

from lib.chat.delivery_dates import normalize_delivery_date
from lib.chat.intent_metadata import IntentMetadata, Vernacular
from lib.chat.off_topic import is_off_topic_message
from lib.neo4j.hybrid_context import extract_budget, extract_max_price

QueryMode = Literal["utility", "situational"]

# Transactional shopping cues — route for speed over empathy.
_UTILITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(search|find|show|list|price|stock|deliver(?:y)?|checkout|cart|track)\b", re.I),
    re.compile(r"\b(cake|flower|chocolate|gift|hamper|bouquet)s?\b", re.I),
    re.compile(r"\b(colombo|kandy|galle|rupees?|rs\.?)\b", re.I),
    re.compile(r"\bVIMP[0-9A-Z]+\b"),
)

# Emotional / life-event cues — favor warmer, local tone (trump utility when matched).
_EMOTIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(broke up|breakup|break-up|passed away|funeral|condolence|sympathy|"
        r"sorry to hear|heartbroken|devastated|grieving)\b",
        re.I,
    ),
)

# Order-fix apologies are transactional — not distress / concierge empathy turns.
_TRANSACTIONAL_APOLOGY = re.compile(
    r"\bi'?m sorry\b.*\b(?:order(?:ed)?|wrong|mistake|replacement|help|fix|cake)\b",
    re.I,
)

_SITUATIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    _EMOTIONAL_PATTERNS
    + (
        re.compile(
            r"\b(missing (?:her|him|them)|lonely|stressed|anxious|nervous)\b",
            re.I,
        ),
        re.compile(r"\b(girlfriend|boyfriend|ex-|divorce|separated)\b", re.I),
        re.compile(
            r"\b(valentine(?:'?s)?|anniversary surprise|romantic surprise|surprise my partner)\b",
            re.I,
        ),
    )
)

# Sinhala script or common Tanglish tokens.
_SINHALA_SCRIPT = re.compile(r"[\u0D80-\u0DFF]")
_TANGLISH_TOKENS: frozenset[str] = frozenset(
    {
        "mage",
        "mama",
        "ammata",
        "aiyo",
        "machan",
        "malli",
        "nangi",
        "eka",
        "ona",
        "karanna",
        "denna",
        "kiyala",
        "hodata",
        "puluvan",
    },
)

_DELIVERY_INTENT = re.compile(
    r"\b(?:deliver(?:y|able)?|ship(?:ping)?|send(?:\s+to)?)\b",
    re.I,
)

_PERISHABLE_GIFT_INTENT = re.compile(
    r"\b(?:cake|cakes|flower|flowers|rose|roses|bouquet|fruit|gift|gifts|hamper|combo)\b",
    re.I,
)

_DELIVER_TO_CITY = re.compile(
    r"\bdeliver(?:y)?\s+(?:to\s+)?(Colombo(?:\s+\d{2})?|Kandy|Galle|Negombo|Jaffna|Matara)\b",
    re.I,
)

_CITY_DELIVERY = re.compile(
    r"\b(Colombo(?:\s+\d{2})?|Kandy|Galle|Negombo|Jaffna)\s+delivery\b",
    re.I,
)

_IN_OR_FOR_CITY = re.compile(
    r"\b(?:in|to|for)\s+(Colombo(?:\s+\d{2})?|Kandy|Galle|Negombo|Jaffna|Matara)\b",
    re.I,
)

_BARE_DELIVERY_CITY = re.compile(
    r"^(Colombo(?:\s+\d{2})?|Kandy|Galle|Negombo|Jaffna|Matara)(?:\s+please)?[.!]?$",
    re.I,
)


def classify_query_mode(text: str) -> QueryMode:
    """Classify input as utility (transactional) or situational (emotional)."""
    stripped = text.strip()
    if not stripped:
        return "utility"

    if _TRANSACTIONAL_APOLOGY.search(stripped):
        return "utility"

    if any(pattern.search(stripped) for pattern in _EMOTIONAL_PATTERNS):
        return "situational"

    situational_hits = sum(1 for pattern in _SITUATIONAL_PATTERNS if pattern.search(stripped))
    utility_hits = sum(1 for pattern in _UTILITY_PATTERNS if pattern.search(stripped))

    if situational_hits > 0 and situational_hits >= utility_hits:
        return "situational"
    return "utility"


def detect_code_switching(text: str) -> bool:
    """True when the message mixes English with Sinhala script or Tanglish."""
    stripped = text.strip()
    if not stripped:
        return False
    if _SINHALA_SCRIPT.search(stripped):
        return True
    tokens = {token.lower() for token in re.findall(r"[A-Za-z']+", stripped)}
    return bool(tokens & _TANGLISH_TOKENS)


def vernacular_score_hint(text: str) -> float:
    """Heuristic 0.0–1.0 hint for local-flavor rubric (not an LLM score)."""
    if not text.strip():
        return 0.0
    score = 0.0
    if _SINHALA_SCRIPT.search(text):
        score += 0.5
    tokens = {token.lower() for token in re.findall(r"[A-Za-z']+", text)}
    overlap = len(tokens & _TANGLISH_TOKENS)
    score += min(0.5, overlap * 0.15)
    return min(1.0, score)


def detect_vernacular(text: str) -> Vernacular:
    """Classify vernacular as English, Sinhala script, or Tanglish."""
    stripped = text.strip()
    if not stripped:
        return "en"
    if _SINHALA_SCRIPT.search(stripped):
        return "si"
    tokens = {token.lower() for token in re.findall(r"[A-Za-z']+", stripped)}
    if tokens & _TANGLISH_TOKENS:
        return "tanglish"
    return "en"


def _has_delivery_intent(text: str) -> bool:
    return bool(_DELIVERY_INTENT.search(text))


def _has_perishable_gift_intent(text: str) -> bool:
    return bool(_PERISHABLE_GIFT_INTENT.search(text))


def _normalize_city(raw: str) -> str:
    parts = raw.strip().split()
    if len(parts) >= 2 and parts[0].lower() == "colombo" and parts[1].isdigit():
        return f"Colombo {parts[1]}"
    return parts[0].capitalize() if parts else raw.strip()


def extract_target_city(text: str) -> str | None:
    """Extract a delivery destination city from delivery verbs or in/to/for city phrases."""
    stripped = text.strip().rstrip(".!")
    bare_match = _BARE_DELIVERY_CITY.match(stripped)
    if bare_match:
        return _normalize_city(bare_match.group(1))
    if _has_delivery_intent(text):
        for pattern in (_DELIVER_TO_CITY, _CITY_DELIVERY):
            match = pattern.search(text)
            if match:
                return _normalize_city(match.group(1))
    match = _IN_OR_FOR_CITY.search(text)
    if match:
        return _normalize_city(match.group(1))
    return None


def is_delivery_context_relevant_turn(
    state: dict[str, object],
    user_message: str,
) -> bool:
    """True when this turn should surface delivery city/date copy."""
    from lib.chat.delivery_dates import normalize_delivery_date

    stripped = user_message.strip()
    intent_metadata = state.get("intent_metadata")
    metadata: dict[str, object] = (
        dict(intent_metadata) if isinstance(intent_metadata, dict) else {}
    )
    if metadata.get("requires_delivery_validation"):
        return True
    if metadata.get("target_city"):
        return True
    if extract_target_city(stripped):
        return True
    if normalize_delivery_date({}, stripped) is not None:
        return True
    if _has_delivery_intent(stripped):
        return True
    if classify_query_mode(stripped) == "situational":
        return False
    return False


class QueryPreprocessor:
    """Derive IntentMetadata from raw user text before LLM intent classification."""

    def process(self, text: str) -> IntentMetadata:
        stripped = text.strip()
        mode = classify_query_mode(stripped)
        vernacular = detect_vernacular(stripped)
        off_topic = is_off_topic_message(stripped)
        target_city = None if off_topic else extract_target_city(stripped)
        has_delivery_date = normalize_delivery_date({}, stripped) is not None
        requires_delivery = (
            not off_topic
            and target_city is not None
            and (
                _has_delivery_intent(stripped)
                or _has_perishable_gift_intent(stripped)
                or has_delivery_date
            )
        )
        budget_cap = extract_budget(stripped)
        return {
            "is_situational": mode == "situational",
            "detected_vernacular": vernacular,
            "requires_delivery_validation": requires_delivery,
            "target_city": target_city,
            "budget_max": budget_cap.amount if budget_cap else extract_max_price(stripped),
            "budget_currency": budget_cap.currency if budget_cap else None,
            "vernacular_score_hint": vernacular_score_hint(stripped),
        }
