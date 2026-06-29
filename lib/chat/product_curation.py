"""Budget-aware sorting and filtering for product carousels."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lib.embeddings.reranker import CrossEncoderService

_NEAR_BUDGET_FACTOR = 1.10
_HIDE_BUDGET_FACTOR = 2.0

_FLOWER_FRUIT_INTENT = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet|floral|fruit|fruits)\b",
    re.I,
)
_PUJA_DENYLIST = re.compile(r"\b(?:puja|pooja|pooj?a|watti|religious)\b", re.I)
_PRODUCE_DENYLIST = re.compile(
    r"\b(?:coconut|banana|grocery|vegetable|potato|onion|tomato|carrot|"
    r"curry powder|kitkat|ritzbury)\b",
    re.I,
)
_GIFT_PROMOTE_RE = re.compile(
    r"\b(?:hamper|hampers|combo|combopack|ferrero|chocolate bouquet|gift box)\b",
    re.I,
)
_GIFT_PROMOTE_WITH_BOUQUET_RE = re.compile(
    r"\b(?:bouquet)\b",
    re.I,
)
_GIFT_DEMOTE_RE = re.compile(
    r"\b(?:curry powder|kitkat|ritzbury|grocery|spice|convenience|"
    r"chocolate bar|toffee bar|candy bar|snack bar|for him|gentleman)\b",
    re.I,
)
_GIFT_VOUCHER_RE = re.compile(r"\b(?:gift\s+)?voucher\b", re.I)
_LOW_TICKET_SNACK_RE = re.compile(r"\b(?:bar|snack|toffee|candy)\b", re.I)
_CATALOG_TYPO_NORMALIZATION: dict[str, str] = {
    "greetting": "greeting",
    "greettings": "greetings",
}
_LOOSE_GROCERY_RE = re.compile(
    r"\b(?:single|loose|fresh)\s+(?:apple|banana|orange|mango|grape|fruit)\b|"
    r"\b(?:candy|toffee|lollipop|mint)\s+(?:pack|bag)?\b",
    re.I,
)
_FEMALE_RECIPIENTS = frozenset(
    {
        "wife",
        "mom",
        "mother",
        "mum",
        "girlfriend",
        "sister",
        "daughter",
        "grandma",
        "grandmother",
    },
)
_MALE_RECIPIENTS = frozenset(
    {
        "husband",
        "dad",
        "father",
        "boyfriend",
        "brother",
        "son",
        "grandpa",
        "grandfather",
    },
)
_NEUTRAL_RECIPIENTS = frozenset(
    {
        "colleague",
        "coworker",
        "co-worker",
        "friend",
        "neighbor",
        "neighbour",
        "boss",
        "teacher",
        "client",
        "customer",
    },
)
_FOR_HIM_RE = re.compile(
    r"\b(?:for him|for dad|father'?s?|men'?s?|gentleman|boyfriend)\b|^Dad\b",
    re.I,
)
_TITLE_LEADING_DAD_RE = re.compile(r"^Dad\b", re.I)
_FOR_HER_RE = re.compile(r"\b(?:for her|for mom|mother'?s?|women'?s?|ladies|girlfriend)\b", re.I)
_NON_FLORAL_FLOWER_DENYLIST = re.compile(
    r"\b(?:air freshener|freshener|fragrance|deodorizer|room spray|scented)\b",
    re.I,
)
_FLORAL_FOR_CHOCOLATE_DENYLIST = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet|floral)\b",
    re.I,
)
_CHOCOLATE_NEGATIVE_CATEGORY_HINTS: tuple[str, ...] = (
    "Flower",
    "Flowers",
    "Bouquet",
    "Floral",
)
_BIRTHDAY_PRODUCT_RE = re.compile(r"\bbirthday\b", re.I)
_CAKE_ID_PREFIX = re.compile(r"^cake", re.I)
_DESSERT_CATEGORY_RE = re.compile(r"\b(?:chocolate|desserts?)\b", re.I)
_GENERIC_DESSERT_RE = re.compile(
    r"\b(?:lava\s+cake|dessert|mousse|brownie|loaf\s+cake|pudding|tiramisu)\b",
    re.I,
)
_CAKE_ACCESSORY_BLACKLIST = re.compile(
    r"\b(?:topper|mould|mold|turning\s+table|cake\s+stand|stand|icing\s+set|"
    r"fondant|nozzle|decorating|piping\s+bag|spatula)\b",
    re.I,
)
_FOCUS_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "chocolate": re.compile(r"\b(?:chocolate|choco|cocoa)\b", re.I),
    "tea": re.compile(r"\b(?:tea|teas|ceylon|dilmah|qualitea)\b", re.I),
    "cake": re.compile(r"\b(?:cake|birthday)\b", re.I),
    "flowers": re.compile(r"\b(?:flower|flowers|rose|roses|bouquet|floral)\b", re.I),
    "gift": re.compile(r"\b(?:hamper|combo|combopack|gift)\b", re.I),
}
_ANNIVERSARY_OCCASION_RE = re.compile(r"\banniversary\b", re.I)
_ANNIVERSARY_PROMOTE_RE = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet|floral|cake|cakes|hamper|hampers|"
    r"chocolate|chocolates|combo|combopack)\b",
    re.I,
)
_ANNIVERSARY_DEMOTE_RE = re.compile(
    r"\b(?:greeting\s+card|watch\s+box|storage\s+box|voucher|gift\s+voucher)\b",
    re.I,
)

PUJA_NEGATIVE_CATEGORY_HINTS: tuple[str, ...] = (
    "Puja",
    "Pooja",
    "Religious offerings",
)


def is_flower_fruit_intent(query: str) -> bool:
    """True when the customer turn targets flowers, bouquets, or fruit gifts."""
    return bool(query.strip() and _FLOWER_FRUIT_INTENT.search(query))


def has_graph_hybrid_context(hybrid_context: dict[str, Any] | None) -> bool:
    """True when GraphRAG produced usable hints or pruned traversal nodes.

    Vector hits alone are insufficient — they do not steer MCP category filters
    when the cross-encoder yields no passing occasion/category hints.
    """
    if not hybrid_context:
        return False
    hints = hybrid_context.get("hints") or {}
    for key in ("occasion", "category", "exclude_categories"):
        value = hints.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return bool(hybrid_context.get("categories") or hybrid_context.get("occasions"))


def product_price_amount(product: dict[str, Any]) -> float | None:
    """Return numeric price amount from a Kapruka search product dict."""
    raw_price = product.get("price")
    if isinstance(raw_price, dict):
        amount = raw_price.get("amount")
        if isinstance(amount, (int, float)):
            return float(amount)
        return None
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    return None


def _normalize_catalog_typos(text: str) -> str:
    normalized = text
    for typo, replacement in _CATALOG_TYPO_NORMALIZATION.items():
        normalized = re.sub(rf"\b{re.escape(typo)}\b", replacement, normalized, flags=re.I)
    return normalized


def _product_text_blob(product: dict[str, Any]) -> str:
    parts = [
        _normalize_catalog_typos(str(product.get("name") or "")),
        _normalize_catalog_typos(str(product.get("summary") or "")),
    ]
    category = product.get("category")
    if isinstance(category, dict):
        parts.extend(str(category.get(key) or "") for key in ("name", "slug", "id"))
    return " ".join(parts)


def product_matches_puja_denylist(product: dict[str, Any]) -> bool:
    """True when product name, summary, or category matches puja/religious denylist."""
    return bool(_PUJA_DENYLIST.search(_product_text_blob(product)))


def product_matches_produce_denylist(product: dict[str, Any]) -> bool:
    """True when product looks like grocery produce rather than a gift."""
    return bool(_PRODUCE_DENYLIST.search(_product_text_blob(product)))


def demote_produce_for_vague_gifts(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Light demotion of obvious produce when the query is a vague gift idea."""
    from lib.chat.intent_heuristics import is_vague_gift_intent

    if not is_vague_gift_intent(query):
        return products
    preferred = [product for product in products if not product_matches_produce_denylist(product)]
    demoted = [product for product in products if product_matches_produce_denylist(product)]
    return preferred + demoted


def _product_category_text(product: dict[str, Any]) -> str:
    category = product.get("category")
    if isinstance(category, dict):
        parts = [str(category.get(key) or "") for key in ("name", "slug", "id")]
        return " ".join(parts)
    return ""


def product_is_birthday_cake_product(product: dict[str, Any]) -> bool:
    """True when Kapruka metadata marks the item as a birthday cake."""
    if _BIRTHDAY_PRODUCT_RE.search(_product_category_text(product)):
        return True
    name = str(product.get("name") or "")
    if _BIRTHDAY_PRODUCT_RE.search(name):
        return True
    product_id = str(product.get("id") or "")
    return bool(_CAKE_ID_PREFIX.match(product_id) and _BIRTHDAY_PRODUCT_RE.search(name))


def product_is_generic_dessert(product: dict[str, Any]) -> bool:
    """True for chocolate/dessert items that are not birthday cakes."""
    if product_is_birthday_cake_product(product):
        return False
    if _DESSERT_CATEGORY_RE.search(_product_category_text(product)):
        return True
    return bool(_GENERIC_DESSERT_RE.search(_product_text_blob(product)))


def is_cake_accessory(product: dict[str, Any]) -> bool:
    """True when a search hit is a cake decorating tool rather than an edible cake."""
    return bool(_CAKE_ACCESSORY_BLACKLIST.search(_product_text_blob(product)))


def filter_cake_accessories(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop baking accessories and decorating tools from cake search results."""
    return [product for product in products if not is_cake_accessory(product)]


def apply_birthday_cake_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    hybrid_context: dict[str, Any] | None = None,
    graph_context_available: bool = False,
    session_product_focus: str | None = None,
    session_flavor_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Prefer birthday-category cakes and demote generic desserts for birthday turns."""
    from lib.neo4j.hybrid_context import is_birthday_cake_scoped_turn

    if not is_birthday_cake_scoped_turn(
        query,
        hybrid_context,
        session_product_focus=session_product_focus,
    ):
        return list(products)

    birthday: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []
    desserts: list[dict[str, Any]] = []
    for product in products:
        if product_is_birthday_cake_product(product):
            birthday.append(product)
        elif product_is_generic_dessert(product):
            desserts.append(product)
        else:
            neutral.append(product)

    if birthday:
        ordered = _boost_chocolate_birthday_cakes(
            birthday + neutral,
            query,
            session_flavor_hint=session_flavor_hint,
        )
        if graph_context_available:
            return ordered + desserts
        return ordered

    if graph_context_available:
        return neutral + desserts
    return neutral


def _boost_chocolate_birthday_cakes(
    products: list[dict[str, Any]],
    query: str,
    *,
    session_flavor_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Promote chocolate birthday cakes when flavor is explicit in the turn."""
    chocolate_focus = session_flavor_hint == "chocolate" or bool(
        _FOCUS_TOKEN_PATTERNS["chocolate"].search(query),
    )
    if not chocolate_focus or not products:
        return list(products)
    preferred = [
        product
        for product in products
        if _FOCUS_TOKEN_PATTERNS["chocolate"].search(_product_text_blob(product))
    ]
    if not preferred:
        return list(products)
    demoted = [product for product in products if product not in preferred]
    return preferred + demoted


def demote_puja_products(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Move puja-denylist items to the end for flower/fruit discovery queries."""
    if not is_flower_fruit_intent(query):
        return list(products)
    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        if product_matches_puja_denylist(product):
            demoted.append(product)
        else:
            preferred.append(product)
    return preferred + demoted


def filter_puja_products(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Drop puja-denylist items for flower/fruit queries when GraphRAG is unavailable."""
    if not is_flower_fruit_intent(query):
        return list(products)
    return [product for product in products if not product_matches_puja_denylist(product)]


def apply_puja_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    graph_context_available: bool,
) -> list[dict[str, Any]]:
    """Demote puja items when graph hints exist; filter them when Neo4j is degraded."""
    if graph_context_available:
        return demote_puja_products(products, query)
    return filter_puja_products(products, query)


def _product_currency(product: dict[str, Any]) -> str:
    raw_price = product.get("price")
    if isinstance(raw_price, dict):
        code = raw_price.get("currency")
        if isinstance(code, str) and code.strip():
            return code.strip().upper()
    return "LKR"


def product_matches_focus(
    product: dict[str, Any],
    session_product_focus: str | None,
) -> bool:
    """True when product name/category matches the session shopping focus."""
    if not session_product_focus:
        return True
    pattern = _FOCUS_TOKEN_PATTERNS.get(session_product_focus)
    if pattern is None:
        return True
    return bool(pattern.search(_product_text_blob(product)))


def carousel_focus_guard(
    products: list[dict[str, Any]],
    session_product_focus: str | None,
    *,
    top_n: int = 5,
    min_ratio: float = 0.3,
) -> bool:
    """True when top carousel items align with session product focus."""
    if not products or not session_product_focus:
        return True
    sample = products[:top_n]
    if not sample:
        return True
    matches = sum(1 for product in sample if product_matches_focus(product, session_product_focus))
    return (matches / len(sample)) >= min_ratio


def demote_off_focus_products(
    products: list[dict[str, Any]],
    session_product_focus: str | None,
) -> list[dict[str, Any]]:
    """Keep in-focus items first; demote off-focus tail instead of dropping all results."""
    if not session_product_focus or not products:
        return list(products)
    matching = [
        product for product in products if product_matches_focus(product, session_product_focus)
    ]
    if not matching:
        return list(products)
    demoted = [
        product for product in products if not product_matches_focus(product, session_product_focus)
    ]
    return matching + demoted


def _gift_curation_active(
    *,
    session_product_focus: str | None,
    user_message: str,
    hybrid_context: dict[str, Any] | None = None,
) -> bool:
    from lib.chat.intent_heuristics import is_budgeted_gift_ideas_message

    if re.search(r"\b(?:tea|teas)\b", user_message, re.I):
        return False
    if is_budgeted_gift_ideas_message(user_message):
        return True
    if session_product_focus == "gift":
        return True
    if session_product_focus == "chocolate":
        return True
    if _BIRTHDAY_PRODUCT_RE.search(user_message):
        return True
    hints = (hybrid_context or {}).get("hints") or {}
    occasion = hints.get("occasion")
    return isinstance(occasion, str) and "birthday" in occasion.lower()


def demote_loose_grocery_items(
    products: list[dict[str, Any]],
    *,
    user_message: str,
) -> list[dict[str, Any]]:
    """Deprioritize single fruit/candy SKUs on budgeted gift-ideas turns."""
    from lib.chat.intent_heuristics import is_budgeted_gift_ideas_message

    if not products or not is_budgeted_gift_ideas_message(user_message):
        return list(products)
    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _LOOSE_GROCERY_RE.search(blob) or (
            _LOW_TICKET_SNACK_RE.search(blob) and not _GIFT_PROMOTE_RE.search(blob)
        ):
            demoted.append(product)
        else:
            preferred.append(product)
    return preferred + demoted


def demote_non_chocolate_for_chocolate_focus(
    products: list[dict[str, Any]],
    query: str,
    *,
    session_product_focus: str | None = None,
) -> list[dict[str, Any]]:
    """Demote floral arrangements when the customer wants chocolate gifts."""
    if not products:
        return []
    chocolate_focus = session_product_focus == "chocolate" or bool(
        _FOCUS_TOKEN_PATTERNS["chocolate"].search(query),
    )
    if not chocolate_focus:
        return list(products)

    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _FLORAL_FOR_CHOCOLATE_DENYLIST.search(blob) and not _FOCUS_TOKEN_PATTERNS[
            "chocolate"
        ].search(blob):
            demoted.append(product)
        else:
            preferred.append(product)
    return preferred + demoted


def demote_non_tea_for_tea_focus(
    products: list[dict[str, Any]],
    query: str,
    *,
    session_product_focus: str | None = None,
) -> list[dict[str, Any]]:
    """Demote biscuits and unrelated gifts when the customer asked for tea."""
    if not products:
        return []
    tea_focus = session_product_focus == "tea" or bool(
        _FOCUS_TOKEN_PATTERNS["tea"].search(query),
    )
    if not tea_focus:
        return list(products)

    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _FOCUS_TOKEN_PATTERNS["tea"].search(blob):
            preferred.append(product)
        else:
            demoted.append(product)
    if not preferred:
        return list(products)
    return preferred + demoted


def demote_non_floral_for_flower_intent(
    products: list[dict[str, Any]],
    query: str,
    *,
    session_product_focus: str | None = None,
) -> list[dict[str, Any]]:
    """Demote air fresheners and room scents when the customer wants fresh flowers."""
    if not products:
        return []
    if not is_flower_fruit_intent(query) and session_product_focus != "flowers":
        return list(products)

    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _NON_FLORAL_FLOWER_DENYLIST.search(blob):
            demoted.append(product)
        else:
            preferred.append(product)
    return preferred + demoted


def boost_carousel_relevance(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Promote exact phrase matches to the front of the carousel."""
    stripped = query.strip()
    if not stripped or len(products) < 2:
        return list(products)

    from graphs.nodes.resolve_cart_product import phrase_product_overlap_score

    scored = [
        (phrase_product_overlap_score(stripped, str(product.get("name") or "")), product)
        for product in products
    ]
    best_score = max(score for score, _ in scored)
    if best_score < 0.6:
        return list(products)
    top = [product for score, product in scored if score == best_score]
    rest = [product for score, product in scored if score < best_score]
    return top + rest


def apply_gift_curation(
    products: list[dict[str, Any]],
    *,
    session_product_focus: str | None = None,
    user_message: str = "",
    hybrid_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Promote hampers/combos; demote grocery/spice/convenience candy on gift turns."""
    if not products or not _gift_curation_active(
        session_product_focus=session_product_focus,
        user_message=user_message,
        hybrid_context=hybrid_context,
    ):
        return list(products)

    user_wants_voucher = bool(_GIFT_VOUCHER_RE.search(user_message))
    promoted: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if not user_wants_voucher and _GIFT_VOUCHER_RE.search(blob) or _GIFT_DEMOTE_RE.search(blob) or _PRODUCE_DENYLIST.search(blob):
            demoted.append(product)
        elif _GIFT_PROMOTE_RE.search(blob) or (
            session_product_focus != "chocolate" and _GIFT_PROMOTE_WITH_BOUQUET_RE.search(blob)
        ):
            promoted.append(product)
        else:
            neutral.append(product)
    curated = promoted + neutral + demoted
    return demote_loose_grocery_items(curated, user_message=user_message)


def filter_gift_noise_products(
    products: list[dict[str, Any]],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    """Drop grocery, spice, and low-ticket snack items on strict budget turns."""
    if not strict or not products:
        return list(products)
    filtered: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _GIFT_DEMOTE_RE.search(blob) or _PRODUCE_DENYLIST.search(blob):
            continue
        price = product_price_amount(product)
        if price is not None and price < 500 and _LOW_TICKET_SNACK_RE.search(blob):
            continue
        filtered.append(product)
    return filtered


def apply_recipient_curation(
    products: list[dict[str, Any]],
    session_recipient_hint: str | None,
) -> list[dict[str, Any]]:
    """Drop gender-mismatched gift sets; fall back to demote-only if fewer than 3 would remain."""
    if not products or not session_recipient_hint:
        return list(products)
    recipient = session_recipient_hint.strip().lower()
    if recipient in _FEMALE_RECIPIENTS:
        mismatch = _FOR_HIM_RE
    elif recipient in _MALE_RECIPIENTS:
        mismatch = _FOR_HER_RE
    elif recipient in _NEUTRAL_RECIPIENTS:
        mismatch = re.compile(
            r"\b(?:for him|for her|for dad|for mom|father'?s?|mother'?s?|"
            r"men'?s?|women'?s?|gentleman|ladies)\b|^Dad\b",
            re.I,
        )
    else:
        return list(products)
    preferred: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        name = str(product.get("name") or "")
        if mismatch.search(blob) or (
            recipient in _FEMALE_RECIPIENTS and _TITLE_LEADING_DAD_RE.search(name)
        ):
            mismatched.append(product)
        else:
            preferred.append(product)
    if len(preferred) >= 3:
        return preferred
    return preferred + mismatched


def _merge_exclude_category_tokens(existing: str, additions: tuple[str, ...]) -> str:
    tokens = [part.strip() for part in existing.split(",") if part.strip()]
    seen = {token.lower() for token in tokens}
    for item in additions:
        if item.lower() not in seen:
            tokens.append(item)
            seen.add(item.lower())
    return ", ".join(tokens)


def _product_matches_excluded_category(product: dict[str, Any], exclude_hints: str) -> bool:
    cat_text = _product_category_text(product).lower()
    blob = _product_text_blob(product).lower()
    for token in exclude_hints.split(","):
        needle = token.strip().lower()
        if needle and (needle in cat_text or needle in blob):
            return True
    return False


def filter_excluded_category_hints(
    products: list[dict[str, Any]],
    hybrid_context: dict[str, Any] | None,
    *,
    session_product_focus: str | None = None,
    query: str = "",
) -> list[dict[str, Any]]:
    """Drop products matching graph exclude_categories hints or chocolate floral noise."""
    if not products:
        return []
    hints = (hybrid_context or {}).get("hints") or {}
    exclude = str(hints.get("exclude_categories") or "")
    chocolate_focus = session_product_focus == "chocolate" or bool(
        _FOCUS_TOKEN_PATTERNS["chocolate"].search(query),
    )
    if chocolate_focus:
        exclude = _merge_exclude_category_tokens(exclude, _CHOCOLATE_NEGATIVE_CATEGORY_HINTS)
    if not exclude.strip():
        return list(products)
    filtered = [
        product for product in products if not _product_matches_excluded_category(product, exclude)
    ]
    if len(filtered) >= 3:
        return filtered
    return list(products)


def refine_last_search_by_budget(
    last_search_products: list[dict[str, Any]],
    *,
    budget_max: float | None,
    currency: str,
    session_product_focus: str | None = None,
    session_search_query: str | None = None,
    session_recipient_hint: str | None = None,
    user_message: str = "",
    hybrid_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Re-filter prior carousel by budget and session focus; None triggers MCP fallback."""
    _ = session_search_query
    if not last_search_products or budget_max is None or budget_max <= 0:
        return None

    curated = curate_carousel_products(
        last_search_products,
        query=user_message or (session_search_query or ""),
        budget_max=budget_max,
        currency=currency,
        hybrid_context=hybrid_context,
        session_product_focus=session_product_focus,
        session_recipient_hint=session_recipient_hint,
        strict_budget=True,
    )
    in_budget = [
        product
        for product in curated
        if (price := product_price_amount(product)) is not None and price <= budget_max
    ]
    if not in_budget:
        return None

    if session_product_focus:
        matching = [
            product
            for product in in_budget
            if product_matches_focus(product, session_product_focus)
        ]
        if not matching:
            return None
        demoted = [
            product
            for product in in_budget
            if not product_matches_focus(product, session_product_focus)
        ]
        if demoted and any(_GIFT_DEMOTE_RE.search(_product_text_blob(item)) for item in demoted):
            return None
        return matching

    return in_budget


def is_anniversary_occasion_intent(
    query: str,
    hybrid_context: dict[str, Any] | None = None,
    session_occasion: str | None = None,
) -> bool:
    """True when the turn targets anniversary gifts (query, session, or graph hint)."""
    if query.strip() and _ANNIVERSARY_OCCASION_RE.search(query):
        return True
    if isinstance(session_occasion, str) and "anniversary" in session_occasion.lower():
        return True
    if hybrid_context:
        hints = hybrid_context.get("hints") or {}
        occasion_hint = str(hints.get("occasion") or "").lower()
        if "anniversary" in occasion_hint:
            return True
    return False


def apply_anniversary_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    hybrid_context: dict[str, Any] | None = None,
    session_occasion: str | None = None,
) -> list[dict[str, Any]]:
    """Drop greeting cards/watch boxes for anniversary turns; promote flowers/hampers."""
    if not is_anniversary_occasion_intent(query, hybrid_context, session_occasion=session_occasion):
        return list(products)

    promoted: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _ANNIVERSARY_DEMOTE_RE.search(blob):
            dropped.append(product)
        elif _ANNIVERSARY_PROMOTE_RE.search(blob):
            promoted.append(product)
        else:
            neutral.append(product)
    kept = promoted + neutral
    if len(kept) >= 3:
        return kept
    return kept + dropped


def sort_and_filter_by_budget(
    products: list[dict[str, Any]],
    budget_max: float | None,
    currency: str,
    *,
    strict_in_budget: bool = False,
) -> list[dict[str, Any]]:
    """Hide items above 2× budget; sort in-budget asc, then near-budget (+10%) with badge."""
    if budget_max is None or budget_max <= 0:
        return list(products)

    target_currency = currency.strip().upper() if currency.strip() else "LKR"
    scoped = [product for product in products if _product_currency(product) == target_currency]
    if not scoped:
        scoped = list(products)

    in_budget: list[dict[str, Any]] = []
    near_budget: list[dict[str, Any]] = []
    over_near: list[dict[str, Any]] = []

    for product in scoped:
        price = product_price_amount(product)
        if price is None:
            if not strict_in_budget:
                over_near.append(product)
            continue
        if strict_in_budget:
            if price <= budget_max:
                in_budget.append(product)
            continue
        if price > budget_max * _HIDE_BUDGET_FACTOR:
            continue
        if price <= budget_max:
            in_budget.append(product)
        elif price <= budget_max * _NEAR_BUDGET_FACTOR:
            tagged = dict(product)
            tagged["slightly_over_budget"] = True
            near_budget.append(tagged)
        else:
            tagged = dict(product)
            tagged["over_budget"] = True
            over_near.append(tagged)

    in_budget.sort(key=lambda item: product_price_amount(item) or 0.0)
    near_budget.sort(key=lambda item: product_price_amount(item) or 0.0)
    over_near.sort(key=lambda item: product_price_amount(item) or 0.0)
    return in_budget + near_budget + over_near


def ensure_flower_price_tier_diversity(
    products: list[dict[str, Any]],
    budget_max: float,
    *,
    top_n: int = 5,
    tier_ratio: float = 0.7,
) -> list[dict[str, Any]]:
    """Ensure at least one sub-tier rose option appears in the top carousel slots."""
    if not products or budget_max <= 0:
        return list(products)
    threshold = budget_max * tier_ratio
    affordable_index: int | None = None
    for index, product in enumerate(products):
        price = product_price_amount(product)
        if price is not None and price < threshold:
            affordable_index = index
            break
    if affordable_index is None or affordable_index < top_n:
        return list(products)
    reordered = list(products)
    affordable = reordered.pop(affordable_index)
    insert_at = min(top_n - 1, len(reordered))
    reordered.insert(insert_at, affordable)
    return reordered


def curate_carousel_products(
    products: list[dict[str, Any]],
    *,
    query: str,
    budget_max: float | None,
    currency: str,
    graph_context_available: bool = False,
    hybrid_context: dict[str, Any] | None = None,
    session_product_focus: str | None = None,
    session_flavor_hint: str | None = None,
    session_recipient_hint: str | None = None,
    session_occasion: str | None = None,
    strict_budget: bool = False,
) -> list[dict[str, Any]]:
    """Apply birthday/puja relevance curation then budget-aware carousel ordering."""
    from lib.neo4j.hybrid_context import is_birthday_cake_scoped_turn

    scoped = apply_birthday_cake_curation(
        products,
        query=query,
        hybrid_context=hybrid_context,
        graph_context_available=graph_context_available,
        session_product_focus=session_product_focus,
        session_flavor_hint=session_flavor_hint,
    )
    scoped = apply_anniversary_curation(
        scoped,
        query=query,
        hybrid_context=hybrid_context,
        session_occasion=session_occasion,
    )
    scoped = apply_gift_curation(
        scoped,
        session_product_focus=session_product_focus,
        user_message=query,
        hybrid_context=hybrid_context,
    )
    scoped = demote_produce_for_vague_gifts(scoped, query)
    scoped = demote_non_floral_for_flower_intent(
        scoped,
        query,
        session_product_focus=session_product_focus,
    )
    scoped = demote_non_chocolate_for_chocolate_focus(
        scoped,
        query,
        session_product_focus=session_product_focus,
    )
    scoped = demote_non_tea_for_tea_focus(
        scoped,
        query,
        session_product_focus=session_product_focus,
    )
    scoped = filter_excluded_category_hints(
        scoped,
        hybrid_context,
        session_product_focus=session_product_focus,
        query=query,
    )
    scoped = demote_off_focus_products(scoped, session_product_focus)
    scoped = apply_recipient_curation(scoped, session_recipient_hint)
    scoped = filter_gift_noise_products(scoped, strict=strict_budget)

    def _budget_sort(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sort_and_filter_by_budget(
            items,
            budget_max,
            currency,
            strict_in_budget=strict_budget,
        )

    def _finalize_carousel(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sorted_items = _budget_sort(items)
        if budget_max is None:
            sorted_items = boost_carousel_relevance(sorted_items, query)
        if budget_max is not None and budget_max > 0 and is_flower_fruit_intent(query):
            sorted_items = ensure_flower_price_tier_diversity(sorted_items, budget_max)
        return sorted_items

    if is_birthday_cake_scoped_turn(
        query,
        hybrid_context,
        session_product_focus=session_product_focus,
    ):
        birthday_items = [
            product for product in scoped if product_is_birthday_cake_product(product)
        ]
        other_items = [
            product for product in scoped if not product_is_birthday_cake_product(product)
        ]
        return _finalize_carousel(birthday_items) + _finalize_carousel(other_items)
    if is_flower_fruit_intent(query):
        if graph_context_available:
            preferred = [
                product for product in scoped if not product_matches_puja_denylist(product)
            ]
            demoted = [product for product in scoped if product_matches_puja_denylist(product)]
            return _finalize_carousel(preferred) + _finalize_carousel(demoted)

        filtered = filter_puja_products(scoped, query)
        return _finalize_carousel(filtered)

    return _finalize_carousel(scoped)


_CARD_SNIPPET_MAX_LEN = 96
_CATALOG_BREADCRUMB_SUMMARY_RE = re.compile(
    r"^(?:specialGifts|cakes|flowers|chocolates|gift)\s*-\s*",
    re.I,
)
_CATEGORY_MARKETING_CRUMB_RE = re.compile(
    r"\b(?:kapruka\s*cakes?(?:\s+cakes)?|celebrate\s+life|special\s+moments)\b",
    re.I,
)
_PRODUCT_ID_WEIGHT_PREFIX_RE = re.compile(
    r"^(?:[A-Za-z]*\d*[A-Z]{2,}\d+[A-Za-z0-9]*)\s+Weight:\s*[\d.]+\s*"
    r"(?:Lbs|Kg|KG)?(?:\s*\([^)]*\))?\s*",
    re.I,
)
_STRAY_OPEN_PAREN_RE = re.compile(r"\(\s+(?=[^)]*$)")


def _looks_like_category_marketing_crumb(text: str) -> bool:
    """True when MCP summary is breadcrumb/marketing copy without product detail."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return False
    if " The " in cleaned:
        return False
    if len(cleaned.split()) < 6:
        return False
    return bool(_CATEGORY_MARKETING_CRUMB_RE.search(cleaned))


def _looks_like_title_case(text: str) -> bool:
    """True when most words after the first are capitalized (MCP Title Case dumps)."""
    words = text.split()
    if len(words) < 3:
        return False
    mid_caps = sum(1 for word in words[1:] if word and word[0].isupper() and word not in {"I"})
    return mid_caps / (len(words) - 1) >= 0.5


def _to_sentence_case(text: str) -> str:
    """Normalize Title Case catalog copy to sentence case."""
    if not text or not _looks_like_title_case(text):
        return text
    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        stripped = sentence.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        sentences.append(lowered[0].upper() + lowered[1:] if lowered else stripped)
    return " ".join(sentences)


def _sanitize_catalog_summary(text: str) -> str:
    """Strip MCP category breadcrumbs and product-id prefixes from card copy."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    the_idx = cleaned.find(" The ")
    if the_idx >= 0:
        cleaned = cleaned[the_idx + 1 :]
    elif _CATALOG_BREADCRUMB_SUMMARY_RE.match(cleaned):
        parts = cleaned.split(" ", 3)
        if len(parts) >= 4:
            cleaned = parts[3]
    cleaned = _PRODUCT_ID_WEIGHT_PREFIX_RE.sub("", cleaned).strip()
    cleaned = re.sub(
        r"^(?:Kapruka Cakes Cakes|Kapruka Cakes)\s+",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    if _looks_like_category_marketing_crumb(cleaned):
        return ""
    return _to_sentence_case(cleaned.strip())


def _sanitize_product_name(name: str) -> str:
    """Normalize catalog product names for carousel display."""
    from lib.utils.text import normalize_catalog_text

    cleaned = normalize_catalog_text(name).replace("`", "")
    cleaned = _STRAY_OPEN_PAREN_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def rerank_products_by_query(
    query: str,
    products: list[dict[str, Any]],
    *,
    reranker: CrossEncoderService | None = None,
) -> list[dict[str, Any]]:
    """Reorder products by cross-encoder relevance to the shopper query."""
    stripped_query = query.strip()
    if not stripped_query or len(products) < 2:
        return list(products)
    from lib.embeddings.reranker import get_reranker

    encoder = reranker or get_reranker()
    texts = [_product_text_blob(product) for product in products]
    scores = encoder.score_pairs(stripped_query, texts)
    ranked = sorted(zip(scores, products, strict=False), key=lambda item: item[0], reverse=True)
    return [product for _, product in ranked]


def _truncate_card_snippet(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= _CARD_SNIPPET_MAX_LEN:
        return cleaned
    clipped = cleaned[:_CARD_SNIPPET_MAX_LEN].rsplit(" ", 1)[0]
    return f"{clipped}…"


def _build_card_description_fallback(
    product: dict[str, Any],
    *,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
    session_delivery_city: str | None = None,  # accepted but not used — causes stale subtitles
) -> str:
    """One-line carousel copy when MCP description is empty."""
    name = str(product.get("name") or "").strip()

    # Product-specific blurbs take priority over generic session context
    if name and _BIRTHDAY_PRODUCT_RE.search(name):
        if _FOCUS_TOKEN_PATTERNS["chocolate"].search(name):
            return "Rich chocolate birthday cake for your celebration."
        return "A celebration cake from Kapruka's curated bakery selection."
    if name and _FOCUS_TOKEN_PATTERNS["flowers"].search(name):
        return "Fresh floral arrangement from Kapruka."
    if name and _FOCUS_TOKEN_PATTERNS["gift"].search(name):
        return "A curated Kapruka gift box or hamper."

    # Occasion/recipient context fallback — delivery city intentionally omitted
    # (including it causes stale "to Galle" subtitles on unrelated turns)
    context_bits: list[str] = []
    if isinstance(session_occasion, str) and session_occasion.strip():
        context_bits.append(f"for {session_occasion.strip()}")
    if isinstance(session_recipient_hint, str) and session_recipient_hint.strip():
        context_bits.append(f"for your {session_recipient_hint.strip()}")

    if context_bits:
        return f"A thoughtful Kapruka pick {' '.join(context_bits)}."

    return "A thoughtful Kapruka gift for your occasion."


def enrich_product_card_description(
    product: dict[str, Any],
    *,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
    session_delivery_city: str | None = None,
) -> dict[str, Any]:
    """Attach card_description_fallback when the catalog omits a description."""
    enriched = dict(product)
    raw_name = str(product.get("name") or "").strip()
    if raw_name:
        enriched["name"] = _sanitize_product_name(raw_name)
    raw_description = str(product.get("description") or "").strip()
    description = _sanitize_catalog_summary(raw_description)
    summary = _sanitize_catalog_summary(str(product.get("summary") or "").strip())
    snippet = description or summary
    if snippet:
        enriched["card_description_fallback"] = _truncate_card_snippet(snippet)
        if raw_description and raw_description != snippet:
            enriched["description"] = ""
        return enriched
    enriched["card_description_fallback"] = _build_card_description_fallback(
        product,
        session_occasion=session_occasion,
        session_recipient_hint=session_recipient_hint,
        session_delivery_city=session_delivery_city,
    )
    if raw_description:
        enriched["description"] = ""
    return enriched


def enrich_carousel_product_descriptions(
    products: list[dict[str, Any]],
    *,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
    session_delivery_city: str | None = None,
) -> list[dict[str, Any]]:
    """Apply occasion-aware card fallbacks across a carousel product list."""
    return [
        enrich_product_card_description(
            product,
            session_occasion=session_occasion,
            session_recipient_hint=session_recipient_hint,
            session_delivery_city=session_delivery_city,
        )
        for product in products
    ]
