# AgenticKapruka Local Chat QA Report

**Date:** 2026-06-30  
**Assessor:** Independent QA (gstack-browse customer dialogue + Kapruka MCP cross-validation)  
**Environment:** `http://127.0.0.1:8080/chat` (local dev)  
**Health at test time:** `healthy` — Redis, Neo4j, neo4j_graphrag, Zep, MCP all up  
**Browse tool:** gstack-browse (`~/.cursor/skills/gstack/browse/dist/browse`)

**STATUS: DONE**

---

## 1. Executive summary

**Overall grade: B (strong shopping concierge on catalog, cart, delivery, and support; product-detail and preference handling regressed)**

Tested as a real shopper through multi-turn dialogue and starter chips: birthday discovery, product detail, delivery fees, ordinal cart adds, support FAQ, off-topic redirect, budget gifts, roses to Galle, order tracking, guest checkout, and proceed-to-checkout. Prices, delivery rates, and order amounts matched Kapruka MCP ground truth with no hallucinations observed.

The agent reads like a professional Kapruka assistant on discovery carousels, verified delivery quotes, cart resolution, order tracking, support handoff, and checkout guidance. The main gaps are product-detail fast-path (weight queries route to generic search), subjective preference handling ("less sweet"), and noisy delivery replies (triplicated perishable warnings).

**What works well**

- Carousel-first discovery for “birthday cake for mom in Colombo” — 9 cakes with Low Stock badges; prices match MCP (Springtime **Rs. 5,770**, Symphony **Rs. 6,500**, etc.).
- Delivery verification: Colombo 05 on 5 July 2026 → **Rs. 300** (MCP: `available: true, rate: 300`).
- Ordinal cart: “add the second one” → **Happy Birthday Symphony Ribbon Cake** at **Rs. 6,500** (MCP `CAKE00KA001827`: 6500).
- Support FAQ: perishable guidance, **+94-11-7551111**, policy URL, clear scope limits.
- Off-topic weather: polite decline + pivot to gifts/delivery.
- Guest checkout: click-to-pay, no account required.
- Budget chip “Gift ideas under Rs. 5,000”: immediate carousel (Tender Love **Rs. 1,990**, Dad Blue Heart **Rs. 2,950** — MCP-confirmed).
- Roses chip “Fresh roses delivery to Galle tomorrow”: rose carousel + **Rs. 1,090** delivery (MCP Galle 2026-07-01: 1090).
- Order tracking `VIMP34456CB2`: structured card, **Delivered**, **Rs. 4,970** (MCP: 4970 LKR), full progress timeline.
- Proceed to checkout: correctly asks for delivery city (Colombo 03, Kandy, Galle).
- No browser console errors during session.

**What needs refinement**

1. **Product detail / weight fast-path broken** — Combined and explicit weight queries (“How much does the Springtime… weigh?”) return a generic carousel or “I don't have information on its weight” despite MCP listing **2.77 Lbs**. Root cause: `_PRODUCT_DETAIL` regex in `lib/chat/product_detail.py` does not match `weight` / `weigh` phrasing.
2. **Subjective preferences ignored** — “not too sweet” on discovery and product follow-up gets no `product_preference_note()` guidance; bot claims missing catalog data instead.
3. **Delivery reply verbosity** — Correct Rs. 300 fee, but perishable warning appears three times (LLM prose + two structured blocks).
4. **Discovery date gate vs carousel** — Opening turn asks for delivery date *and* shows a full carousel; does not acknowledge sweetness preference.
5. **Carousel subtitle noise** — Breadcrumb fragments remain (“Birthday Kapruka Cakes Celebrate Life s Special Moments…”).
6. **Roses chip latency** — “Putting together recommendations…” visible ~20s before carousel; total chip response ~34s.
7. **Cart drawer UX** — Composer is editable with drawer open (fixed), but Send button stays disabled until drawer closes.
8. **New Session retains cart** — Cart count persists across “New Session” (Symphony cake still in cart).
9. **Neo4j ontology incomplete** — Startup warnings: missing `Product` label and `CO_PURCHASED_WITH` relationship; hybrid GraphRAG enrichment may be degraded despite `/health` reporting up.

With weight-regex fix, preference routing on combined detail turns, and delivery-warning deduplication, this is close to production-polished for discovery → cart → guest click-to-pay.

---

## 2. Test matrix

| # | Scenario | User message / action | Bot behavior | MCP validation | Result |
|---|----------|----------------------|--------------|----------------|--------|
| 1 | Product discovery | “birthday cake for mom in Colombo… elegant, not too sweet” | Date gate + 9-item cake carousel; no sweetness guidance | `kapruka_search_products` → Springtime **5770**, Symphony **6500** | **PASS** (preference gap) |
| 2 | Product detail | “Springtime… weight and less sweet?” | “I don't have information on its weight or sweetness”; re-shows carousel | `kapruka_get_product` CAKE00KA001685 → weight **2.77**, price **5770** | **FAIL** |
| 3 | Delivery fee | “Colombo 05 this Saturday, July 5th — delivery fee?” | Rs. **300** + Saturday/Sunday clarification; perishable warning ×3 | `kapruka_check_delivery` Colombo 05, 2026-07-05 → rate **300** | **PASS** (noisy) |
| 4 | Ordinal cart | “Please add the second one to my cart” | Added **Happy Birthday Symphony Ribbon Cake** | `CAKE00KA001827` → **6500** | **PASS** |
| 5 | Support / FAQ | “return and refund policy if cake arrives damaged?” | Perishable guidance, phone, policy URL, scope limits | N/A | **PASS** |
| 6 | Off-topic | “What's the weather in Colombo today?” | Polite decline + pivot to gifts | N/A | **PASS** |
| 7 | Guest checkout | “Can I still checkout as a guest?” | Click-to-pay, no login, Proceed to checkout path | N/A | **PASS** |
| 8 | Budget chip | “Gift ideas under Rs. 5,000” | Carousel ≤ Rs. 5,000; prose cites Tender Love **1,990** | `Tender Love Chocolate Box` → **1990** | **PASS** |
| 9 | Roses chip | “Fresh roses delivery to Galle tomorrow” | Rose carousel + delivery **Rs. 1,090**; slow load | Galle 2026-07-01 → rate **1090** | **PASS** (slow) |
| 10 | Track order | “Track order VIMP34456CB2” | Structured tracking card, Delivered, Rs. **4,970** | MCP track → **4970** LKR, delivered | **PASS** |
| 11 | Explicit weight | “How much does the Springtime Birthday Ribbon Cake weigh?” | Generic carousel; no weight cited | MCP weight **2.77 Lbs** | **FAIL** |
| 12 | Proceed to checkout | “I want to proceed to checkout” | Asks delivery city | N/A | **PASS** |
| 13 | Cart drawer | Open cart while chatting | Composer editable; Send disabled | N/A | **PARTIAL** |

---

## 3. MCP cross-validation detail

| Claim | Agent said | MCP ground truth | Match |
|-------|-----------|------------------|-------|
| Springtime cake price | Rs. 5,770 | 5770 LKR | ✓ |
| Symphony cake (cart) | Rs. 6,500 | 6500 LKR | ✓ |
| Colombo 05 delivery (5 Jul) | Rs. 300 | rate 300, available | ✓ |
| Galle delivery (1 Jul) | Rs. 1,090 | rate 1090, available | ✓ |
| Tender Love chocolate | Rs. 1,990 | 1990 LKR | ✓ |
| Order VIMP34456CB2 amount | Rs. 4,970 | 4970 LKR | ✓ |
| Springtime weight | “don't have information” | 2.77 Lbs (1.25 KG) | ✗ |

---

## 4. Persona notes (authentic customer lens)

**Tone:** Warm and professional. The bot avoids jargon, cites Kapruka verification for delivery, and sets expectations on perishables. Support and off-topic boundaries feel trustworthy.

**Friction points a real customer would notice:**

- Asked about cake sweetness twice; got “I don't have that information” instead of honest catalog limits plus a human suggestion (e.g. smaller portion, message Kapruka support).
- Delivery fee answer buried under repeated warnings.
- Weight question on a product already in the carousel should be instant; waiting ~12s for another carousel feels broken.
- Roses chip spinner (“Putting together recommendations…”) long enough to wonder if the app hung.

**Delight moments:**

- “Added Happy Birthday Symphony Ribbon Cake to your cart” — correct ordinal resolution without re-asking.
- Order tracking card with live-tracking note and timeline — feels like a real concierge pulling from Kapruka systems.
- Guest checkout explanation is clear and reduces anxiety for overseas senders.

---

## 5. Technical observations

- **Health:** All five services up including `mcp` and `neo4j_graphrag`.
- **Neo4j warnings at startup:** `UnknownLabelWarning: Product`, `UnknownRelationshipTypeWarning: CO_PURCHASED_WITH`. Recommend `python scripts/bootstrap_neo4j.py` before evals.
- **Product detail routing:** `is_product_detail_turn()` matches `tell me about`, `ingredients`, etc., but not `\bweight\b` / `\bweigh\b` / `\bhow much does.*weigh`. Weight formatting exists in `product_weight()` and `build_product_detail_reply()` but is never reached for natural phrasing.
- **Console:** No JS errors; no SSE failures this session (improvement vs 2026-06-28 report).
- **Rate limits:** Not exercised (60 req/min MCP cap); prior friendly rate-limit copy not re-tested.

---

## 6. Recommended fixes (priority order)

| Priority | Issue | Suggested fix |
|----------|-------|---------------|
| P0 | Weight queries miss fast-path | Extend `_PRODUCT_DETAIL` regex with `\b(?:weight|weigh|how much does.*weigh)\b`; resolve product by name from `last_visible_products` |
| P0 | Combined detail turn denies catalog data | When product name matches carousel + MCP `get_product` has weight, never say “don't have information”; call `resolve_product_detail` |
| P1 | Sweetness / preference | Route `_PREFERENCE_SWEETNESS_RE` matches through `product_preference_note()` even on LLM synthesis path |
| P1 | Delivery warning dedup | Emit perishable warning once in `generate_response` delivery fast-path |
| P2 | Discovery opener | Acknowledge preference; either gate carousel on date or skip date ask when city+weekend already given |
| P2 | Carousel subtitles | Verify `_sanitize_catalog_summary()` runs on current carousel path |
| P3 | Roses chip latency | Profile hybrid context + MCP for flower intents; consider chip-specific fast-path |
| P3 | New Session cart | Clear or confirm cart on session reset |

---

## 7. Comparison to 2026-06-29 re-QA

| Area | 2026-06-29 | 2026-06-30 (this run) |
|------|------------|------------------------|
| Budget chip carousel | PASS | PASS |
| Delivery fee (no product dump) | PASS | PASS (still verbose warnings) |
| Product detail weight | PASS (explicit query) | **FAIL** (explicit + combined) |
| Less-sweet preference | PARTIAL | **FAIL** (denies data) |
| Cart drawer composer | PASS | PASS (Send still disabled) |
| Console SSE errors | Not noted | None observed |

Regression on product-detail weight is the highest-risk finding for this branch.

---

## 8. Test plan for re-verification

- [ ] “How much does the Springtime Birthday Ribbon Cake weigh?” → “2.77 Lbs” without carousel
- [ ] Combined weight + sweetness → weight from catalog + honesty on sweetness
- [ ] Delivery fee only → single perishable warning max
- [ ] Roses chip → carousel within 15s
- [ ] New Session → cart policy documented or cleared
- [ ] MCP rate-limit exhaustion → friendly retry copy (manual)
