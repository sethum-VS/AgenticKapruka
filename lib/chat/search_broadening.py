"""Search query broadening ladder when Kapruka MCP returns empty results."""

from __future__ import annotations

import re
from typing import Any, Literal

from lib.chat.intent_metadata import IntentMetadata
from lib.neo4j.hybrid_context import strip_location_from_search_query

BroadenStep = Literal["gift_voucher_fallback", "simplify_q", "strip_city", "drop_max_price"]

BROADEN_LADDER: tuple[BroadenStep, ...] = (
    "simplify_q",
    "strip_city",
    "gift_voucher_fallback",
    "drop_max_price",
)

_GIFT_IN_Q = re.compile(r"\bgifts?\b", re.I)

_SIMPLIFY_Q_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbirthday\s+cake\b", re.I), "cake"),
    (re.compile(r"\banniversary\s+cake\b", re.I), "cake"),
    (re.compile(r"\bwedding\s+cake\b", re.I), "cake"),
    (re.compile(r"\bbirthday\s+gift\b", re.I), "gift"),
)


def _normalize_q(q: str) -> str:
    return re.sub(r"\s{2,}", " ", q).strip(" ,.-")


def broaden_search_args(
    args: dict[str, Any],
    step: BroadenStep,
    *,
    intent_metadata: IntentMetadata | None = None,
) -> dict[str, Any] | None:
    """Apply one ladder step to kapruka_search_products args.

    Returns a new args dict when the step changes the query, or None when it does not apply.
    """
    q = str(args.get("q") or "").strip()
    if not q:
        return None

    if step == "gift_voucher_fallback":
        if re.search(r"\btea\b", q, re.I):
            simplified = "tea" if q.lower().strip() != "tea" else None
            if simplified:
                return {**args, "q": simplified}
            return None
        if not _GIFT_IN_Q.search(q):
            return None
        if q.lower().strip() in {"voucher", "gift voucher"}:
            return None
        if re.search(
            r"\b(?:hamper|chocolate|flower|flowers|cake|roses?|bouquet|combo)\b",
            q,
            re.I,
        ):
            return None
        return {**args, "q": "voucher"}

    if step == "simplify_q":
        if re.search(r"\btea\b", q, re.I) and re.search(r"\bgift\b", q, re.I):
            return {**args, "q": "tea"}
        if str(args.get("category") or "").strip().lower() == "birthday":
            return None
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
        new_q = strip_location_from_search_query(q, intent_metadata)
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


def first_applicable_broaden_step(
    args: dict[str, Any],
    *,
    preserve_max_price: bool = False,
    intent_metadata: IntentMetadata | None = None,
) -> BroadenStep | None:
    """Return the first ladder step that would change the given search args."""
    for step in BROADEN_LADDER:
        if step == "drop_max_price" and preserve_max_price:
            continue
        if broaden_search_args(args, step, intent_metadata=intent_metadata) is not None:
            return step
    return None


def apply_first_broaden(
    args: dict[str, Any],
    *,
    preserve_max_price: bool = False,
    intent_metadata: IntentMetadata | None = None,
) -> tuple[dict[str, Any] | None, BroadenStep | None]:
    """Apply the first applicable broaden step; at most one step per call."""
    step = first_applicable_broaden_step(
        args,
        preserve_max_price=preserve_max_price,
        intent_metadata=intent_metadata,
    )
    if step is None:
        return None, None
    broadened = broaden_search_args(args, step, intent_metadata=intent_metadata)
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
