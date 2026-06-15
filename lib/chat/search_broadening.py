"""Search query broadening ladder when Kapruka MCP returns empty results."""

from __future__ import annotations

import re
from typing import Any, Literal

from lib.neo4j.hybrid_context import strip_location_from_search_query

BroadenStep = Literal["simplify_q", "strip_city", "drop_max_price"]

BROADEN_LADDER: tuple[BroadenStep, ...] = (
    "simplify_q",
    "strip_city",
    "drop_max_price",
)

_SIMPLIFY_Q_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbirthday\s+cake\b", re.I), "cake"),
    (re.compile(r"\banniversary\s+cake\b", re.I), "cake"),
    (re.compile(r"\bwedding\s+cake\b", re.I), "cake"),
)


def _normalize_q(q: str) -> str:
    return re.sub(r"\s{2,}", " ", q).strip(" ,.-")


def broaden_search_args(args: dict[str, Any], step: BroadenStep) -> dict[str, Any] | None:
    """Apply one ladder step to kapruka_search_products args.

    Returns a new args dict when the step changes the query, or None when it does not apply.
    """
    q = str(args.get("q") or "").strip()
    if not q:
        return None

    if step == "simplify_q":
        new_q = q
        changed = False
        for pattern, replacement in _SIMPLIFY_Q_REPLACEMENTS:
            replaced, count = pattern.subn(replacement, new_q)
            if count:
                new_q = _normalize_q(replaced)
                changed = True
        if not changed or not new_q or new_q == q:
            return None
        return {**args, "q": new_q}

    if step == "strip_city":
        new_q = strip_location_from_search_query(q)
        if not new_q or new_q == q:
            return None
        return {**args, "q": new_q}

    if step == "drop_max_price":
        if "max_price" not in args:
            return None
        updated = dict(args)
        updated.pop("max_price", None)
        return updated

    return None


def first_applicable_broaden_step(args: dict[str, Any]) -> BroadenStep | None:
    """Return the first ladder step that would change the given search args."""
    for step in BROADEN_LADDER:
        if broaden_search_args(args, step) is not None:
            return step
    return None


def apply_first_broaden(
    args: dict[str, Any],
) -> tuple[dict[str, Any] | None, BroadenStep | None]:
    """Apply the first applicable broaden step; at most one step per call."""
    step = first_applicable_broaden_step(args)
    if step is None:
        return None, None
    broadened = broaden_search_args(args, step)
    return broadened, step


def build_empty_search_reply(*, broaden_attempted: bool = False) -> str:
    """Customer-facing copy when discovery search returns no products."""
    if broaden_attempted:
        return (
            "I broadened the search but still couldn't find a close match on Kapruka. "
            "Try a simpler gift type such as cake, flowers, or chocolates, "
            "a slightly higher budget, or a nearby delivery city — and I can search again."
        )
    return (
        "I couldn't find products matching that exact search on Kapruka. "
        "Try a broader gift type such as cake, flowers, or chocolates, "
        "or share your budget and delivery city and I'll search again."
    )
