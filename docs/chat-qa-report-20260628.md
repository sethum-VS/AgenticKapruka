# AgenticKapruka Local Chat QA Report

**Date:** 2026-06-28  
**Assessor:** Independent QA (gstack-browse customer dialogue + Kapruka MCP cross-validation)  
**Environment:** `http://127.0.0.1:8080/chat` (local dev, branch `refactor/cart-drawer-components`)  
**Health at test time:** `healthy` after brief 503 at session start — Redis, Neo4j, neo4j_graphrag, Zep, MCP all up

**STATUS: DONE_WITH_CONCERNS**

---

## 1. Executive summary

**Overall grade: B+ (strong core shopping flows, presentation and rate-limit handling need polish)**

Tested as a real shopper through a single multi-turn session: birthday discovery, product detail, delivery fees, ordinal cart adds, support FAQ, off-topic redirect, budget gifts (starter chip + natural phrasing), guest checkout guidance, and proceed-to-checkout. The agent reads like a professional Kapruka concierge on catalog, delivery, cart, and support paths. Prices and delivery rates matched Kapruka MCP ground truth with no hallucinations observed.

**What works well**

- Carousel-first discovery for “birthday cake for mom in Colombo” — 9+ cakes with Low Stock badges, Colombo delivery note, no bureaucratic date gate.
- Product detail for Springtime cake: ID `cake00KA001685`, **Rs. 5,770**, weight **2.77 Lbs** — matches MCP exactly.
- Delivery verification: “Sunday, 28 June 2026… **Rs. 300**” for Colombo 05 — MCP confirms `available: true, rate: 300`.
- Ordinal cart resolution: “add the second one” → **Happy Birthday Symphony Ribbon Cake** at **Rs. 6,500** (MCP: 6500).
- Support FAQ handoff: perishable guidance, **+94-11-7551111**, policy URL, clear “shopping assistant only” scope.
- Off-topic weather redirect without breaking rapport.
- Guest checkout explanation: click-to-pay, no account, Proceed to checkout path.
- Proceed to checkout correctly asks for delivery city (Colombo 03, Kandy, Galle).

**What still needs refinement**

1. **Delivery-only queries bundle redundant product content** — fee answer correct, but preceded by Springtime product dump and followed by cake carousel.
2. **Product detail ignores subjective preference** — asked about “less sweet”; bot returned raw catalog text, no dietary guidance.
4. **MCP rate-limit UX** — after ~15 turns, “chocolate gift for my wife, budget 5000” returned a technical error (“Rate limit exceeded”) instead of a customer-friendly retry or cached fallback.
5. **Cart drawer blocks chat input** — browse `fill` timeouts when drawer open; currency selector accidentally clicked during automation.
6. **Turn latency** — discovery ~5s (fast); product detail and delivery ~10s; budget query waited until rate-limit error surfaced.
7. **Console SSE blip** — one `chat SSE stream failed TypeError: network error` during session.

With delivery-only reply cleanup, budget chip fast-path restored, and softer rate-limit messaging, this is close to production-polished for discovery → cart → guest click-to-pay.

---

## 2. Test matrix

| # | Scenario | User message / action | Bot behavior | MCP validation | Result |
|---|----------|----------------------|--------------|----------------|--------|
| 1 | Product discovery | “birthday cake for mom in Colombo… elegant, not too sweet” | Immediate 9-item cake carousel; Colombo delivery note | `kapruka_search_products` → Springtime **5770**, Symphony **6500** | **PASS** |
| 2 | Product detail | “Tell me more about Springtime… weight and less sweet?” | ID, weight **2.77 Lbs**, price **Rs. 5,770**; no sweetness guidance | `kapruka_get_product` CAKE00KA001685 → **5770**, weight **2.77** | **PASS** (preference unanswered) |
| 3 | Delivery fee | “Colombo 05 this Sunday — delivery fee?” | “Sunday, 28 June 2026: **Rs. 300** (verified with Kapruka)” + redundant carousel | `kapruka_check_delivery` Colombo 05, 2026-06-28 → rate **300** | **PASS** (noisy response) |
| 4 | Ordinal cart | “Please add the second one to my cart” | “Added **Happy Birthday Symphony Ribbon Cake** to your cart.” Cart: **Rs. 6,500** | `CAKE00KA001827` → **6500** | **PASS** |
| 5 | Support / FAQ | “return and refund policy if cake arrives damaged?” | Perishable guidance, phone, policy URL, scope limits | N/A | **PASS** |
| 6 | Off-topic | “What's the weather in Colombo today?” | Polite decline + pivot to gifts/delivery | N/A | **PASS** |
| 7a | Budget gift (starter) | Clicked “Gift ideas under Rs. 5,000” | Clarifying question only — no carousel | N/A | **PARTIAL FAIL** (regression vs 06-27) |
| 7b | Budget gift (natural) | “chocolate gift for my wife, budget around 5000 rupees” | “Rate limit exceeded. Wait a moment before retrying.” | MCP has chocolates ≤5000 (e.g. Java I Love You **5000**) | **FAIL** (rate limit) |
| 8 | Guest checkout info | “Can I checkout as a guest without creating an account?” | Clear click-to-pay path; Proceed to checkout; no login required | N/A | **PASS** |
| 9 | Proceed checkout | Cart drawer **Proceed to checkout** | “Which Kapruka delivery city… Colombo 03, Kandy, or Galle.” | N/A | **PASS** (city step only) |

**Note:** Cursor `CallMcpTool` to `user-kapruka` timed out repeatedly. Live validation succeeded via project `.venv` + `lib/kapruka/mcp_client.MCPHttpClient` against `https://mcp.kapruka.com/mcp`.

---

## 3. Strengths

- **Catalog fidelity:** Verified product prices match Kapruka MCP (`CAKE00KA001685` → 5770, `CAKE00KA001827` → 6500). No price hallucinations observed.
- **Delivery professionalism:** Parses “this Sunday” to **28 June 2026**, canonical zone **Colombo 05**, flat fee **Rs. 300** with explicit “verified with Kapruka” attribution.
- **Ordinal cart resolution:** “The second one” maps to the second carousel item after discovery.
- **Support boundaries:** Refund FAQ deflects to Kapruka support; does not pretend to process refunds.
- **Off-topic handling:** Weather redirected without breaking rapport.
- **Guest path clarity:** Guest checkout explanation aligns with Kapruka guest click-to-pay model (60-minute locked-price link).
- **Infrastructure:** Neo4j GraphRAG up — rich birthday cake carousel despite hybrid ranking path; `master_flow` node active in traces.
- **Response speed on discovery:** First carousel appeared in ~5s (improved vs prior ~10–20s reports).

---

## 4. Gaps & refinements (prioritized)

### P0 — High customer impact

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Rate-limit error surfaces to customer | Backend: `KaprukaError: Rate limit exceeded`; user sees technical message | `lib/kapruka/errors.py`, `graphs/nodes/agent_loop.py` — friendly retry copy; backoff; serve cached carousel when possible |


### P1 — Important refinements

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Delivery-only answers bundle carousels | Delivery fee correct but preceded by product dump + cake list | `lib/chat/routing.py` / delivery-only fast-path — answer fee only |
| Product preference questions ignored | “less sweet” unanswered; raw MCP description pasted | `generate_response.py` — synthesize answer from attributes; admit limits honestly |
| Cart drawer blocks chat input | Browse `fill` timeouts when drawer open | `static/js/cart-drawer.js` — close on outside click; don’t cover composer |
| Generic carousel subtitles | Truncated category breadcrumbs as card copy | `lib/chat/product_curation.py` — use MCP `summary` snippets |

### P2 — Polish

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| SSE network error in console | `chat SSE stream failed TypeError: network error` | Investigate stream reconnect / proxy timeout |
| Health 503 blip at cold start | `/health` returned 503 before browse session | Neo4j connection warmup / readiness probe delay |
| Currency selector mis-click risk | Automation clicked EUR option adjacent to send | Wider hit targets / separate composer from header controls |

---

## 5. MCP alignment

| Bot claim | MCP ground truth (`MCPHttpClient`) | Match? |
|-----------|--------------------------------------|--------|
| Springtime Birthday Ribbon Cake Rs. 5,770 (`CAKE00KA001685`) | `price.amount: 5770`, weight `2.77` | ✅ |
| Happy Birthday Symphony Ribbon Cake Rs. 6,500 | `CAKE00KA001827` → 6500, weight 2.44 | ✅ |
| Colombo 05, 2026-06-28, Rs. 300 delivery | `available: true, rate: 300` | ✅ |
| Chocolate gifts ≤ Rs. 5,000 (not shown due to rate limit) | Java I Love You Dark Slab **5000**, Java Still You 15pc **5000** | ⚠️ Catalog exists; bot could not surface |
| MCP bare search `birthday cake` | Returns Springtime + Symphony with prices | ✅ (hybrid path aligned this session) |

**No price hallucinations observed** in this session.

**MCP limits respected during QA:** ~15+ tool calls in ~8 minutes triggered upstream rate limit (60 req/min per IP). Did not invoke `kapruka_create_order` (30/hour guest limit).

---

## 6. UX observations

| Area | Observation |
|------|-------------|
| **Latency** | Discovery ~5s; product detail ~5s; delivery ~10s; budget query failed after extended wait |
| **Carousel** | Rich grid with images, stock badges, Add to cart; subtitles are truncated catalog breadcrumbs |
| **Clarifying questions** | Discovery no longer date-gated (good); budget chip now over-clarifies (regression) |
| **Cart drawer** | Line items, quantity stepper, proceed button work; blocks composer when open |
| **Checkout** | Proceed correctly enters `delivery_city` step; guest click-to-pay not exercised end-to-end |
| **Starter chips** | Birthday cake chip path excellent; budget chip needs fast-path restore |
| **Guest mode** | “Kapruka Guest” label + currency selector clear |
| **Errors** | Rate-limit message is too technical for shoppers |

---

## 7. Recommendations

1. **Soften rate-limit handling** — customer-facing retry message; exponential backoff; optional Redis-cached last-good carousel.
2. **Restore budget chip fast-path** — “Gift ideas under Rs. 5,000” should show curated carousel immediately (as in 06-27 QA).
3. **Lean delivery-only replies** — fee + date + availability only; skip product carousel unless user asks.
4. **Answer preference questions** — when user asks “less sweet,” synthesize from product description or state honestly that sweetness isn’t specified.
5. **Cart drawer UX** — auto-close or don’t obscure message input after Proceed.
6. **Add regression tests** for: budget starter chip carousel, rate-limit graceful degradation, delivery-only (no carousel).
7. **Re-run guest checkout E2E** through click-to-pay link creation (`kapruka_create_order` — respect 30/hour guest limit) in a separate low-traffic session.

---

## Appendix: tooling notes

- **Browse:** `~/.cursor/skills/gstack/browse/dist/browse` (`$B`)
- **MCP validation:** `.venv` + `MCPHttpClient.connect()` (Cursor `CallMcpTool` timed out)
- **Health:** `curl -fsS http://localhost:8080/health`
- **Logs:** `.dev/backend.log` — `master_flow`, `agent_loop` exit_reason `tool_error` on rate limit
- **Screenshot:** `/tmp/chat-qa-20260628.png` (guest checkout response)
- **MCP rate limits:** 60 req/min per IP; `kapruka_create_order` 30/hour; 30-min read cache; guest click-to-pay 60-minute locked prices
