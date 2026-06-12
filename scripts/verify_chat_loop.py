#!/usr/bin/env python3
"""Multi-turn chat verification loop against a running dev server.

Usage:
  make dev   # terminal 1
  make logs  # terminal 2 (optional)
  python scripts/verify_chat_loop.py [--base-url http://localhost:8080]

Sends diverse prompts in one session and reports pass/fail per scenario.
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
        name="planning_task",
        message="help me plan a surprise anniversary dinner gift under 10000 LKR",
        expect_carousel=False,
    ),
    TurnScenario(
        name="specific_product",
        message="chocolate birthday cake",
        expect_carousel=True,
    ),
    TurnScenario(
        name="assistance_request",
        message="what delivery cities do you support?",
        expect_carousel=False,
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
) -> tuple[str, str | None]:
    url = f"{base_url.rstrip('/')}/chat/stream"
    data = parse.urlencode({"message": message}).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if cookie_header:
        headers["Cookie"] = cookie_header

    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=SSE_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            set_cookie = resp.headers.get("Set-Cookie")
            return body, set_cookie
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


def _evaluate_turn(scenario: TurnScenario, html: str) -> list[str]:
    failures: list[str] = []
    lower = html.lower()
    has_carousel = 'data-testid="product-carousel"' in html

    if scenario.expect_carousel and not has_carousel:
        failures.append("expected product carousel, none found")
    if not scenario.expect_carousel and has_carousel:
        failures.append("unexpected product carousel")

    if scenario.expect_clarifying:
        clarifying_markers = ("?", "more detail", "narrow", "which", "what kind")
        if not any(marker in lower for marker in clarifying_markers):
            failures.append("expected clarifying follow-up text")

    for forbidden in scenario.forbidden_substrings:
        if forbidden.lower() in lower:
            failures.append(f"forbidden substring present: {forbidden!r}")

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
            body, set_cookie = _post_chat(base_url, scenario.message, cookie_header)
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
            print(f"  PASS — carousel={'yes' if carousel else 'no'}")

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
