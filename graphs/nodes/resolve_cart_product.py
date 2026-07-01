"""Resolve add-to-cart utterances against prior search results or MCP search."""

from __future__ import annotations

import logging
import re
from typing import Any

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.intent_heuristics import extract_cart_product_phrase
from lib.chat.product_curation import _sanitize_product_name
from lib.chat.product_reference import (
    _normalize_ordinal_phrase,
    is_deictic_phrase,
    is_ordinal_phrase,
    resolve_product_intent_for_cart,
    resolve_product_reference,
)
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.utils.text import normalize_catalog_text, normalize_for_product_match

logger = logging.getLogger(__name__)

_OVERLAP_THRESHOLD = 0.6
_BORDERLINE_OVERLAP_THRESHOLD = 0.4
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STOP_WORDS = frozenset({"the", "a", "an", "to", "my", "please", "add", "put", "in", "into"})
_ARTICLE_PREFIX = re.compile(r"^(?:a|an|the)\s+", re.I)


def _tokenize_for_overlap(text: str) -> set[str]:
    normalized = normalize_for_product_match(text)
    tokens = {token.lower() for token in _TOKEN_RE.findall(normalized)}
    return tokens - _STOP_WORDS


def phrase_product_overlap_score(phrase: str, product_name: str) -> float:
    """Return fraction of phrase tokens found in the product name (0.0–1.0)."""
    phrase_tokens = _tokenize_for_overlap(phrase)
    if not phrase_tokens:
        return 0.0
    name_tokens = _tokenize_for_overlap(product_name)
    if not name_tokens:
        return 0.0
    return len(phrase_tokens & name_tokens) / len(phrase_tokens)


def _product_name(product: dict[str, Any]) -> str:
    name = product.get("name")
    if name is None:
        return ""
    return normalize_catalog_text(str(name))


def match_products_by_phrase(
    phrase: str,
    products: list[dict[str, Any]],
    *,
    threshold: float = _OVERLAP_THRESHOLD,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """Match phrase to products; return winner, tied candidates, or clarifying question."""
    if not phrase.strip() or not products:
        return None, [], None

    scored: list[tuple[float, dict[str, Any]]] = []
    for product in products:
        score = phrase_product_overlap_score(phrase, _product_name(product))
        if score >= threshold:
            scored.append((score, product))

    if not scored:
        if threshold > _BORDERLINE_OVERLAP_THRESHOLD:
            return match_products_by_phrase(
                phrase,
                products,
                threshold=_BORDERLINE_OVERLAP_THRESHOLD,
            )
        return None, [], None

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0]
    top_matches = [product for score, product in scored if score == top_score]

    if len(top_matches) == 1:
        return top_matches[0], [], None

    order = {id(product): index for index, product in enumerate(products)}
    top_matches.sort(key=lambda product: order.get(id(product), len(products)))

    names = ", ".join(
        f"'{_sanitize_product_name(_product_name(product))}'" for product in top_matches[:3]
    )
    question = f"I found a few matches for {phrase!r}. Which one should I add — {names}?"
    return None, top_matches, question


def _search_result_products(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("error"):
        return []
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return []
    products = [item for item in raw_results if isinstance(item, dict)]
    for product in products:
        name = product.get("name")
        if isinstance(name, str):
            product["name"] = normalize_catalog_text(name)
    return products


async def resolve_cart_product(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: resolve which catalog product the customer wants in cart."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    phrase = extract_cart_product_phrase(user_message)
    if not phrase:
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": ("Which product would you like me to add to your cart?"),
            },
        }

    last_visible = list(state.get("last_visible_products") or [])
    last_search = list(state.get("last_search_products") or [])
    reference = resolve_product_reference(
        phrase,
        last_visible_products=last_visible or None,
        last_search_products=last_search or None,
        session_product_focus=state.get("session_product_focus"),
    )
    if reference is not None:
        if reference.get("status") == "resolved" and reference.get("product") is not None:
            resolved_product = reference["product"] or {}
            logger.info(
                "resolve_cart_product: reference %r -> %s",
                phrase,
                resolved_product.get("id"),
            )
            return {
                "cart_action_result": {
                    "status": "resolved",
                    "product": resolved_product,
                    "phrase": phrase,
                },
            }
        clarify_question = reference.get("clarifying_question")
        if isinstance(clarify_question, str) and clarify_question.strip():
            payload: dict[str, Any] = {
                "status": "clarify",
                "product": None,
                "clarifying_question": clarify_question.strip(),
            }
            candidates = reference.get("candidates")
            if isinstance(candidates, list) and candidates:
                payload["candidates"] = candidates
            return {"cart_action_result": payload}

    product, tied, clarify = match_products_by_phrase(phrase, last_search)
    if product is None and not clarify and last_search:
        product, tied, clarify = match_products_by_phrase(
            phrase,
            last_search,
            threshold=_BORDERLINE_OVERLAP_THRESHOLD,
        )
    # Fallback: also search last_visible_products (carousel may show products from merged searches)
    if product is None and not clarify and last_visible:
        product, tied, clarify = match_products_by_phrase(phrase, last_visible)
        if product is None and not clarify:
            product, tied, clarify = match_products_by_phrase(
                phrase,
                last_visible,
                threshold=_BORDERLINE_OVERLAP_THRESHOLD,
            )
    if clarify:
        logger.info("resolve_cart_product: tie among %d candidates for %r", len(tied), phrase)
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": clarify,
                "candidates": tied,
            },
        }
    if product is not None:
        logger.info(
            "resolve_cart_product: matched %r to %s from last_search_products",
            phrase,
            product.get("id"),
        )
        return {
            "cart_action_result": {
                "status": "resolved",
                "product": product,
                "phrase": phrase,
            },
        }

    if is_deictic_phrase(phrase) or is_ordinal_phrase(_normalize_ordinal_phrase(phrase)):
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": ("Search for a gift first, then say 'add that to my cart'."),
            },
        }

    if kapruka_service is None:
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": (
                    f"I couldn't find {phrase!r} in your recent search. "
                    "Try searching for it first, or name the product more specifically."
                ),
            },
        }

    currency = state.get("currency") or "LKR"
    search_query = _ARTICLE_PREFIX.sub("", phrase.strip())
    search_output = await kapruka_service.search_products(
        client_ip or "127.0.0.1",
        q=search_query,
        currency=currency,
        limit=10,
    )
    search_dict = search_output.model_dump(mode="json")
    cold_products = _search_result_products(search_dict)
    budget_raw = state.get("budget_max")
    budget_max = float(budget_raw) if isinstance(budget_raw, (int, float)) else None
    ranked_products = resolve_product_intent_for_cart(
        user_message,
        cold_products,
        search_phrase=search_query,
        session_product_focus=state.get("session_product_focus"),
        hybrid_context=state.get("hybrid_context"),
        currency=currency,
        budget_max=budget_max,
    )
    product, tied, clarify = match_products_by_phrase(search_query, ranked_products)
    if clarify:
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": clarify,
                "candidates": tied,
            },
            "last_search_products": ranked_products,
            "tool_results": {SEARCH_PRODUCTS_TOOL: search_dict},
        }
    if product is not None:
        return {
            "cart_action_result": {
                "status": "resolved",
                "product": product,
                "phrase": phrase,
            },
            "last_search_products": ranked_products,
            "tool_results": {SEARCH_PRODUCTS_TOOL: search_dict},
        }

    return {
        "cart_action_result": {
            "status": "clarify",
            "product": None,
            "clarifying_question": (
                f"I couldn't find a product matching {phrase!r} on Kapruka. "
                "Try a different name or search for gifts first."
            ),
        },
        "last_search_products": ranked_products,
        "tool_results": {SEARCH_PRODUCTS_TOOL: search_dict},
    }
