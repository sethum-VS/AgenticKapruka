#!/usr/bin/env python3
"""Multi-turn chat verification loop against a running dev server.

Usage:
  make dev   # terminal 1
  make logs  # terminal 2 (optional)
  python scripts/verify_chat_loop.py [--base-url http://localhost:8080]

Sends diverse prompts in one session and reports pass/fail per scenario.

Expected TTHW (time-to-helpful-widget) on local dev with mocked or live MCP:
  greeting            ~2–5s   static welcome, no carousel
  broad_gifts         ~3–8s   clarifying question (ask_user), no carousel
  cakes_after_clarify ~8–20s  product carousel after prior clarify turn
  category_flowers    ~8–20s  product carousel
  specific_product    ~8–20s  product carousel
  tracking_order      ~5–15s  order-tracking-status card, no carousel
  tracking_ka         ~5–15s  KA legacy educate copy, no tracking card
  tracking_status     ~5–15s  check-status phrasing + VIMP tracking card
  delivery_colombo    ~8–25s  delivery confirmation or clarifying date, no carousel
  budget_sort         ~8–20s  carousel first item within stated budget cap
  silk_disclaimer     ~8–20s  artificial floral note when silk products appear
  farewell            ~2–5s   warm sign-off, not capabilities menu
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from dataclasses import dataclass
from urllib import error, parse, request

DEFAULT_BASE_URL = "http://localhost:8080"
SSE_TIMEOUT_S = 120.0
TURN_PAUSE_S = 1.5


@dataclass(frozen=True, slots=True)
class TurnScenario:
    name: str
    message: str
    expect_carousel: bool
    expect_clarifying: bool = False
    expect_tracking: bool = False
    expect_tracking_educate: bool = False
    expect_delivery: bool = False
    max_first_carousel_price: float | None = None
    expect_artificial_disclaimer_if_silk: bool = False
    expect_farewell: bool = False
    forbidden_substrings: tuple[str, ...] = ()


SCENARIOS: tuple[TurnScenario, ...] = (
    TurnScenario(
        name="greeting",
        message="hello",
        expect_carousel=False,
    ),
    TurnScenario(
        name="broad_gifts",
        message="show me some gifts",
        expect_carousel=False,
        expect_clarifying=True,
        forbidden_substrings=("couldn't find products",),
    ),
    TurnScenario(
        name="cakes_after_clarify",
        message="cakes",
        expect_carousel=True,
        forbidden_substrings=("couldn't find products", "previous search for 'gifts'"),
    ),
    TurnScenario(
        name="category_flowers",
        message="show me flowers for a birthday",
        expect_carousel=True,
    ),
    TurnScenario(
        name="specific_product",
        message="chocolate birthday cake",
        expect_carousel=True,
    ),
    TurnScenario(
        name="tracking_order",
        message="Track order VIMP34456CB2",
        expect_carousel=False,
        expect_tracking=True,
    ),
    TurnScenario(
        name="tracking_ka",
        message="Where is order KA123456?",
        expect_carousel=False,
        expect_tracking=False,
        expect_tracking_educate=True,
    ),
    TurnScenario(
        name="tracking_status",
        message="check status of my order VIMP34456CB2",
        expect_carousel=False,
        expect_tracking=True,
    ),
    TurnScenario(
        name="delivery_colombo",
        message="can you deliver flowers to Colombo next Saturday?",
        expect_carousel=False,
        expect_delivery=True,
    ),
    TurnScenario(
        name="budget_sort",
        message="wife birthday chocolate flowers ~8000 LKR colombo",
        expect_carousel=True,
        max_first_carousel_price=8000.0,
    ),
    TurnScenario(
        name="silk_disclaimer",
        message="chocolate and flowers wife birthday",
        expect_carousel=True,
        expect_artificial_disclaimer_if_silk=True,
    ),
    TurnScenario(
        name="farewell",
        message="thanks that's all",
        expect_carousel=False,
        expect_farewell=True,
        forbidden_substrings=(
            "Welcome to Kapruka",
            "What would you like to explore",
            "I can help you with:",
        ),
    ),
)


def _parse_sse_assistant_html(body: str) -> str:
    chunks: list[str] = []
    for block in body.split("\n\n"):
        if not block.startswith("event: message"):
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                chunks.append(line.removeprefix("data: "))
    return "\n".join(chunks)


def _post_chat(
    base_url: str,
    message: str,
    cookie_header: str | None,
) -> tuple[str, str | None, float]:
    url = f"{base_url.rstrip('/')}/chat/stream"
    data = parse.urlencode({"message": message}).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if cookie_header:
        headers["Cookie"] = cookie_header

    req = request.Request(url, data=data, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=SSE_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            set_cookie = resp.headers.get("Set-Cookie")
            return body, set_cookie, time.monotonic() - started
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code} for {message!r}: {detail[:500]}"
        raise RuntimeError(msg) from exc
    except error.URLError as exc:
        msg = f"Cannot reach {base_url}: {exc}"
        raise RuntimeError(msg) from exc


def _session_cookie_from_set_cookie(set_cookie: str | None) -> str | None:
    if not set_cookie:
        return None
    match = re.search(r"ak_session=([^;]+)", set_cookie)
    if not match:
        return None
    return f"ak_session={match.group(1)}"


def _extract_first_carousel_price(html: str) -> float | None:
    carousel_idx = html.find('data-testid="product-carousel"')
    if carousel_idx < 0:
        return None
    fragment = html[carousel_idx:]
    match = re.search(
        r'data-testid="product-price"[^>]*>\s*(?:Rs\.\s*|[$£]|A\$|C\$|€)?([\d,]+(?:\.\d+)?)',
        fragment,
    )
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _evaluate_turn(scenario: TurnScenario, html: str) -> list[str]:
    failures: list[str] = []
    lower = html.lower()
    has_carousel = 'data-testid="product-carousel"' in html
    has_tracking = 'data-testid="order-tracking-status"' in html

    if scenario.expect_carousel and not has_carousel:
        failures.append("expected product carousel, none found")
    if not scenario.expect_carousel and has_carousel:
        failures.append("unexpected product carousel")

    if scenario.max_first_carousel_price is not None and has_carousel:
        first_price = _extract_first_carousel_price(html)
        if first_price is None:
            failures.append("could not parse first carousel item price")
        elif first_price > scenario.max_first_carousel_price:
            failures.append(
                f"first carousel item price {first_price:.0f} exceeds budget "
                f"{scenario.max_first_carousel_price:.0f}",
            )

    if scenario.expect_clarifying:
        clarifying_markers = (
            "?",
            "more detail",
            "narrow",
            "which",
            "what kind",
            "who",
            "occasion",
            "recipient",
        )
        if not any(marker in lower for marker in clarifying_markers):
            failures.append("expected clarifying follow-up text")

    if scenario.expect_tracking and not has_tracking:
        failures.append("expected order tracking status card")

    if scenario.expect_tracking_educate:
        educate_markers = (
            "vimp",
            "post-payment",
            "confirmation email",
            "legacy",
        )
        if not any(marker in lower for marker in educate_markers):
            failures.append("expected KA legacy tracking educate copy")
        if has_tracking:
            failures.append("unexpected tracking card for KA legacy educate path")

    if scenario.expect_delivery:
        delivery_markers = (
            "deliver",
            "delivery",
            "colombo",
            "saturday",
            "when would you like",
            "delivery date",
        )
        if not any(marker in lower for marker in delivery_markers):
            failures.append("expected delivery-related response text")

    if scenario.expect_farewell:
        farewell_markers = (
            "you're very welcome",
            "take care",
            "lovely helping you",
        )
        if not any(marker in lower for marker in farewell_markers):
            failures.append("expected warm farewell sign-off")

    for forbidden in scenario.forbidden_substrings:
        if forbidden.lower() in lower:
            failures.append(f"forbidden substring present: {forbidden!r}")

    if scenario.expect_artificial_disclaimer_if_silk:
        has_silk_pick = bool(re.search(r"\bsilk\b", html, re.I)) and bool(
            re.search(r"\b(?:rose|roses|flower|flowers|bouquet)\b", html, re.I),
        )
        disclaimer_markers = (
            "artificial",
            "not fresh-cut",
            "not fresh cut",
            "silk or artificial",
        )
        if has_silk_pick and not any(marker in lower for marker in disclaimer_markers):
            failures.append("expected artificial floral disclaimer when silk products in results")

    if 'role="alert"' in html and "Something went wrong" in html:
        failures.append("error banner in response")

    return failures


def run_loop(base_url: str) -> int:
    cookie_header: str | None = None
    passed = 0
    failed = 0

    print(f"Chat verification loop → {base_url}")
    print(f"Session trace id: verify-{uuid.uuid4().hex[:8]}")
    print("-" * 60)

    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"[{index}/{len(SCENARIOS)}] {scenario.name}: {scenario.message!r}")
        try:
            body, set_cookie, elapsed_s = _post_chat(base_url, scenario.message, cookie_header)
        except RuntimeError as exc:
            print(f"  FAIL — {exc}")
            failed += 1
            continue

        if set_cookie:
            cookie_header = _session_cookie_from_set_cookie(set_cookie) or cookie_header

        html = _parse_sse_assistant_html(body)
        turn_failures = _evaluate_turn(scenario, html)

        if turn_failures:
            failed += 1
            for item in turn_failures:
                print(f"  FAIL — {item}")
            snippet = re.sub(r"\s+", " ", html)[:240]
            print(f"  snippet: {snippet}...")
        else:
            passed += 1
            carousel = 'data-testid="product-carousel"' in html
            tracking = 'data-testid="order-tracking-status"' in html
            print(
                f"  PASS — {elapsed_s:.1f}s carousel={'yes' if carousel else 'no'}"
                f" tracking={'yes' if tracking else 'no'}"
            )

        time.sleep(TURN_PAUSE_S)

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify multi-turn chat against dev server")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    return run_loop(args.base_url)


if __name__ == "__main__":
    sys.exit(main())
