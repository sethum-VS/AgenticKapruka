"""Resolve add-to-cart utterances against prior search results or MCP search."""

from __future__ import annotations

import logging
import re
from typing import Any

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.intent_heuristics import extract_cart_product_phrase
from lib.chat.product_reference import (
    is_deictic_phrase,
    is_ordinal_phrase,
    resolve_product_reference,
)
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL

logger = logging.getLogger(__name__)

_OVERLAP_THRESHOLD = 0.6
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STOP_WORDS = frozenset({"the", "a", "an", "to", "my", "please", "add", "put", "in", "into"})


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens = {token.lower() for token in _TOKEN_RE.findall(text)}
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
    return str(name) if name is not None else ""


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
        return None, [], None

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0]
    top_matches = [product for score, product in scored if score == top_score]

    if len(top_matches) == 1:
        return top_matches[0], [], None

    names = ", ".join(f"'{_product_name(product)}'" for product in top_matches[:3])
    question = f"I found a few matches for {phrase!r}. Which one should I add — {names}?"
    return None, top_matches, question


def _search_result_products(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("error"):
        return []
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return []
    return [item for item in raw_results if isinstance(item, dict)]


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
            product = reference["product"]
            logger.info(
                "resolve_cart_product: reference %r -> %s",
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

    if is_deictic_phrase(phrase) or is_ordinal_phrase(phrase):
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": (
                    "Search for a gift first, then say 'add that to my cart'."
                ),
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
    search_output = await kapruka_service.search_products(
        client_ip or "127.0.0.1",
        q=phrase,
        currency=currency,
        limit=10,
    )
    search_dict = search_output.model_dump(mode="json")
    cold_products = _search_result_products(search_dict)
    product, tied, clarify = match_products_by_phrase(phrase, cold_products)
    if clarify:
        return {
            "cart_action_result": {
                "status": "clarify",
                "product": None,
                "clarifying_question": clarify,
                "candidates": tied,
            },
            "last_search_products": cold_products,
            "tool_results": {SEARCH_PRODUCTS_TOOL: search_dict},
        }
    if product is not None:
        return {
            "cart_action_result": {
                "status": "resolved",
                "product": product,
                "phrase": phrase,
            },
            "last_search_products": cold_products,
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
        "last_search_products": cold_products,
        "tool_results": {SEARCH_PRODUCTS_TOOL: search_dict},
    }
