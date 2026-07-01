"""Keyword heuristics for intent routing when Gemini is unavailable."""

from __future__ import annotations

import re
from typing import Literal

from lib.chat.off_topic import is_off_topic_message
from lib.chat.support_faq import is_support_question
from lib.checkout.tracking import KA_LEGACY_RE, ORD_REF_RE, VIMP_RE
from lib.neo4j.hybrid_context import extract_budget, extract_max_price

Intent = Literal["discovery", "checkout", "tracking", "general", "cart"]

_CART_ADD_TO_PATTERN = re.compile(
    r"\badd\s+(.+?)\s+to\s+(?:my\s+)?cart\b",
    re.I,
)
_CART_PUT_IN_PATTERN = re.compile(
    r"\bput\s+(.+?)\s+in(?:to)?\s+(?:my\s+)?cart\b",
    re.I,
)
# Bare "add [product]" without explicit "to cart" — anchored to avoid false positives
_CART_ADD_BARE_PATTERN = re.compile(
    r"^(?:please\s+)?add\s+(?:the\s+)?(.+?)(?:\s+(?:please|now|for\s+me))?\s*[.!?]?\s*$",
    re.I,
)

_TRACKING_GUARD_TOKENS: frozenset[str] = frozenset(
    (
        "track",
        "track my order",
        "where is my order",
        "order status",
        "check status",
        "shipped",
    ),
)

_PLACE_ORDER_RE = re.compile(
    r"\bplace\s+(?:my|the|an)?\s*order\b",
    re.I,
)

_CHECKOUT_TRIGGER_TOKENS: frozenset[str] = frozenset(
    (
        "checkout",
        "check out",
        "place my order",
        "place the order",
        "place an order",
        "place order",
        "help me place",
        "my cart",
        "view cart",
        "pay now",
        "complete my order",
        "proceed to payment",
        "recipient details",
        "sender details",
    ),
)

PROCEED_CHECKOUT_MESSAGE = "Proceed to checkout"

_GUEST_CHECKOUT_RE = re.compile(
    r"\b(?:checkout|check\s+out)\s+as\s+(?:a\s+)?guest\b|"
    r"\bguest\s+(?:checkout|check\s*out)\b|"
    r"\b(?:can|could)\s+i\s+(?:checkout|check\s+out|order|pay)\s+(?:as\s+(?:a\s+)?guest|without\s+(?:an?\s+)?account)\b",
    re.I,
)


def is_guest_checkout_question(message: str) -> bool:
    """True for informational guest-checkout questions — not a checkout trigger."""
    return bool(message.strip() and _GUEST_CHECKOUT_RE.search(message.strip()))


def build_guest_checkout_reply(*, cart_has_items: bool = True) -> str:
    """Explain Kapruka click-to-pay guest checkout."""
    if cart_has_items:
        lead = (
            "Yes — you can checkout as a guest on Kapruka without creating an account. "
            "Use Proceed to checkout (or ask me to start checkout), then complete delivery "
            "details in chat and pay via the secure Kapruka click-to-pay link."
        )
    else:
        lead = (
            "Yes — Kapruka supports guest checkout (click-to-pay) without an account. "
            "Add a gift to your cart first, then say Proceed to checkout to continue."
        )
    return (
        f"{lead}\n\n"
        "You'll enter recipient and delivery details during checkout; no login is required "
        "to place a guest order."
    )


_VAGUE_GIFT_RE = re.compile(
    r"\b(?:gift ideas|present ideas|what should i gift)\b",
    re.I,
)
_GIFT_SPECIFIC_RE = re.compile(
    r"\b(?:cake|flower|hamper|voucher|mom|dad|mother|father|birthday|anniversary|chocolate|roses?)\b",
    re.I,
)

GIFT_PREFERENCES_QUESTION = (
    "Who is the gift for, and do you have a style in mind — flowers, cake, voucher, or hamper? "
    "For example: 'birthday cake for mom under Rs 5,000'."
)


def is_vague_gift_intent(message: str) -> bool:
    """True for broad gift-idea queries without occasion, recipient, or product type."""
    stripped = message.strip()
    if not stripped or not _VAGUE_GIFT_RE.search(stripped):
        return False
    if extract_budget(stripped) is not None:
        return False
    return not _GIFT_SPECIFIC_RE.search(stripped)


def is_budgeted_gift_ideas_message(message: str) -> bool:
    """True for actionable gift-idea chips that include an explicit budget."""
    stripped = message.strip()
    if not stripped or not _VAGUE_GIFT_RE.search(stripped):
        return False
    return extract_budget(stripped) is not None


_BUDGET_GIFT_RECIPIENT_RE = re.compile(
    r"\b(?:wife|husband|mom|mother|mum|dad|father|girlfriend|boyfriend|partner|"
    r"sister|brother|son|daughter|grandma|grandmother|grandpa|grandfather|"
    r"colleague|friend|boss)\b",
    re.I,
)


def is_natural_budget_gift_message(message: str) -> bool:
    """Budget + recipient or gift phrasing without an explicit product category."""
    stripped = message.strip()
    if not stripped or is_off_topic_message(stripped):
        return False
    if extract_budget(stripped) is None and extract_max_price(stripped) is None:
        return False
    if is_budgeted_gift_ideas_message(stripped):
        return True
    if _PRODUCT_CATEGORY_TOKENS.search(stripped):
        return False
    return bool(_BUDGET_GIFT_RECIPIENT_RE.search(stripped) or _VAGUE_GIFT_RE.search(stripped))


def is_cart_add_trigger(message: str) -> bool:
    """Return True for add-to-cart or put-in-cart phrasing."""
    text = message.strip()
    if not text:
        return False
    return bool(
        _CART_ADD_TO_PATTERN.search(text)
        or _CART_PUT_IN_PATTERN.search(text)
        or _CART_ADD_BARE_PATTERN.match(text)
    )


def extract_cart_product_phrase(message: str) -> str | None:
    """Extract the product phrase from an add-to-cart utterance."""
    text = message.strip()
    if not text:
        return None
    for pattern in (_CART_ADD_TO_PATTERN, _CART_PUT_IN_PATTERN):
        match = pattern.search(text)
        if match:
            phrase = match.group(1).strip(" .,!?:;\"'")
            return phrase or None
    # Bare "add [product]" without "to cart" — lower-priority fallback
    m = _CART_ADD_BARE_PATTERN.match(text)
    if m:
        phrase = m.group(1).strip(" .,!?:;\"'")
        return phrase or None
    return None


def is_proceed_checkout_message(message: str) -> bool:
    """Return True for the exact cart-drawer proceed-to-checkout trigger."""
    return message.strip() == PROCEED_CHECKOUT_MESSAGE


def is_tracking_guard(message: str) -> bool:
    """Return True when message matches tracking fast-path (order number or track keywords)."""
    lowered = message.strip().lower()
    if not lowered:
        return False
    if VIMP_RE.search(message) or KA_LEGACY_RE.search(message) or ORD_REF_RE.search(message):
        return True
    return any(token in lowered for token in _TRACKING_GUARD_TOKENS)


def is_order_intent_message(message: str) -> bool:
    """Return True when the shopper wants to finalize or place an order."""
    text = message.strip()
    if not text:
        return False
    if _PLACE_ORDER_RE.search(text):
        return True
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "place the order",
            "place an order",
            "place my order",
            "help me place",
            "complete my order",
            "proceed to payment",
        )
    )


def is_checkout_trigger(message: str) -> bool:
    """Return True for explicit checkout/cart-view triggers — not add-to-cart phrases."""
    lowered = message.strip().lower()
    if not lowered:
        return False
    if is_guest_checkout_question(message):
        return False
    if is_cart_add_trigger(message):
        return False
    if is_proceed_checkout_message(message):
        return True
    if is_order_intent_message(message):
        return True
    return any(token in lowered for token in _CHECKOUT_TRIGGER_TOKENS)


def classify_routing_guard(message: str) -> Intent | None:
    """Shared guard ordering: cart_add → proceed_checkout → tracking → checkout view."""
    if is_cart_add_trigger(message):
        return "cart"
    if is_proceed_checkout_message(message):
        return "checkout"
    if is_tracking_guard(message):
        return "tracking"
    if is_checkout_trigger(message):
        return "checkout"
    return None


_PRODUCT_CATEGORY_TOKENS = re.compile(
    r"\b(?:cake|cupcakes?|flower|flowers|rose|roses|bouquet|chocolate|chocolates|"
    r"hamper|voucher|gift|gifts|combo|combopack)\b",
    re.I,
)
_TOPIC_PIVOT_PREFIX = re.compile(
    r"^(?:never\s*mind|nevermind|instead|actually|what\s+about)\b",
    re.I,
)
_BARE_CATEGORY_REPLY = re.compile(
    r"^(?:cakes?|flowers?|chocolates?|roses?|bouquets?|gifts?)\s*[!.?]*$",
    re.I,
)


def has_explicit_budget_constraint(
    message: str,
    _session_budget: float | None = None,
    *,
    topic_pivot: bool = False,
) -> bool:
    """True when the turn carries an explicit budget cap (strict carousel filter mode)."""
    if topic_pivot:
        return False
    stripped = message.strip()
    if not stripped:
        return False
    if extract_budget(stripped) is not None or extract_max_price(stripped) is not None:
        return True
    if is_budget_refinement_message(stripped):
        return True
    return is_budgeted_gift_ideas_message(stripped)


def is_budget_refinement_message(message: str) -> bool:
    """True when the turn states a budget without naming a new product category."""
    stripped = message.strip()
    if not stripped or is_off_topic_message(stripped):
        return False
    has_budget = extract_budget(stripped) is not None or extract_max_price(stripped) is not None
    if not has_budget:
        return False
    return not _PRODUCT_CATEGORY_TOKENS.search(stripped)


def is_topic_pivot_message(message: str) -> bool:
    """True when the customer abandons the prior topic for a new product category."""
    stripped = message.strip()
    if not stripped:
        return False
    if _TOPIC_PIVOT_PREFIX.search(stripped):
        return True
    if re.search(r"\binstead\b", stripped, re.I):
        return True
    return bool(re.search(r"nevermind.*\b(?:cakes?|flowers?|chocolates?)\b", stripped, re.I))


_CATEGORY_BROWSE_TOKENS: tuple[str, ...] = (
    "categories",
    "kinds of gifts",
    "what can i buy",
    "what do you sell",
)


def is_category_browse_message(message: str) -> bool:
    """True when the shopper asks to browse Kapruka's category tree, not search products."""
    lowered = message.strip().lower()
    return any(token in lowered for token in _CATEGORY_BROWSE_TOKENS)


def is_bare_category_pivot(message: str) -> str | None:
    """Return the bare category noun when a pivot is category-only (no occasion in turn)."""
    stripped = message.strip().strip("!.?")
    if not stripped or not is_topic_pivot_message(message):
        return None
    if re.search(r"\b(?:birthday|anniversary|wedding|valentine)\b", stripped, re.I):
        return None
    if re.search(r"\b(?:for|under|below|mom|dad|wife|husband)\b", stripped, re.I):
        return None
    lowered = stripped.lower()
    if re.search(r"\b(?:cup)?cakes?\b", lowered):
        return "cake"
    if re.search(r"\b(?:flower|flowers|rose|roses|bouquet)s?\b", lowered):
        return "flowers"
    if re.search(r"\b(?:chocolate|chocolates)\b", lowered):
        return "chocolate"
    if re.search(r"\bgifts?\b", lowered):
        return "gift"
    return None


def infer_intent_from_message(message: str) -> Intent:
    """Map a user utterance to a shopping-graph intent without calling Gemini."""
    guard = classify_routing_guard(message)
    if guard is not None:
        return guard

    lowered = message.strip().lower()
    if not lowered:
        return "general"

    if is_support_question(message):
        return "general"

    if is_order_intent_message(message):
        return "checkout"

    if any(
        token in lowered
        for token in (
            "checkout",
            "deliver",
            "delivery",
            "pay",
            "recipient",
            "sender",
        )
    ):
        return "checkout"

    if is_category_browse_message(message):
        return "general"

    if lowered.startswith("cake00ka") or "product " in lowered:
        return "discovery"

    return "discovery"
