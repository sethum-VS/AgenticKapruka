"""Off-topic and impossible-catalog detectors for shopping-graph routing."""

from __future__ import annotations

import re

_OFF_TOPIC_WEATHER = re.compile(
    r"\b(?:weather|forecast|temperature|rain|rainy|sunny|humidity|cyclone)\b",
    re.I,
)
_OFF_TOPIC_NEWS = re.compile(
    r"\b(?:news|headlines|election|politics|stock\s+market)\b",
    re.I,
)
_OFF_TOPIC_SPORTS = re.compile(
    r"\b(?:cricket|football|rugby|score|match\s+result|world\s+cup)\b",
    re.I,
)
_OFF_TOPIC_MATH = re.compile(
    r"^(?:what\s+is\s+)?\d+\s*[\+\-\*\/]\s*\d+",
    re.I,
)

_IMPOSSIBLE_LIVE_ANIMAL = re.compile(
    r"\b(?:live|real)\s+(?:elephant|tiger|lion|whale|dolphin|puppy|kitten|snake)\b",
    re.I,
)
_IMPOSSIBLE_ELEPHANT = re.compile(r"\b(?:elephant|elephants)\b", re.I)
_LIVE_MODIFIER = re.compile(r"\b(?:live|real|actual)\b", re.I)


def is_off_topic_message(message: str) -> bool:
    """True for weather, news, sports, or general-knowledge turns outside Kapruka scope."""
    stripped = message.strip()
    if not stripped:
        return False
    if _OFF_TOPIC_WEATHER.search(stripped):
        return True
    if _OFF_TOPIC_NEWS.search(stripped):
        return True
    if _OFF_TOPIC_SPORTS.search(stripped):
        return True
    return bool(_OFF_TOPIC_MATH.match(stripped))


def off_topic_topic(message: str) -> str:
    """Return a short topic label for redirect copy."""
    stripped = message.strip()
    if _OFF_TOPIC_WEATHER.search(stripped):
        return "weather"
    if _OFF_TOPIC_NEWS.search(stripped):
        return "news"
    if _OFF_TOPIC_SPORTS.search(stripped):
        return "sports"
    if _OFF_TOPIC_MATH.match(stripped):
        return "math"
    return "that"


def is_impossible_catalog_request(message: str) -> bool:
    """True when the customer asks for live animals or other non-catalog items."""
    stripped = message.strip()
    if not stripped:
        return False
    if _IMPOSSIBLE_LIVE_ANIMAL.search(stripped):
        return True
    return bool(_LIVE_MODIFIER.search(stripped) and _IMPOSSIBLE_ELEPHANT.search(stripped))


def impossible_request_subject(message: str) -> str:
    """Extract a short subject for impossible-request redirect copy."""
    if _IMPOSSIBLE_ELEPHANT.search(message):
        return "live elephant"
    match = _IMPOSSIBLE_LIVE_ANIMAL.search(message)
    if match:
        return match.group(0).strip().lower()
    return "that item"
