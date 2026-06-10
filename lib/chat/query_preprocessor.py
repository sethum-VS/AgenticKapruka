"""Pre-graph query signals: utility vs situational mode and vernacular hints."""

from __future__ import annotations

import re
from typing import Literal

QueryMode = Literal["utility", "situational"]

# Transactional shopping cues — route for speed over empathy.
_UTILITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(search|find|show|list|price|stock|deliver(?:y)?|checkout|cart|track)\b", re.I),
    re.compile(r"\b(cake|flower|chocolate|gift|hamper|bouquet)s?\b", re.I),
    re.compile(r"\b(colombo|kandy|galle|rupees?|rs\.?)\b", re.I),
    re.compile(r"\bVIMP[0-9A-Z]+\b"),
)

# Emotional / life-event cues — favor warmer, local tone.
_SITUATIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(broke up|breakup|passed away|funeral|condolence|sorry|heartbroken|"
        r"missing (?:her|him|them)|devastated|lonely|stressed|anxious)\b",
        re.I,
    ),
    re.compile(r"\b(girlfriend|boyfriend|ex-|divorce|separated)\b", re.I),
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


def classify_query_mode(text: str) -> QueryMode:
    """Classify input as utility (transactional) or situational (emotional)."""
    stripped = text.strip()
    if not stripped:
        return "utility"

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
