# AgenticKapruka Chat QA Report

**Date:** 2026-06-29  
**Assessor:** Independent QA (gstack-browse customer dialogue + Kapruka MCP cross-validation)  
**Target:** `https://agentic-kapruka-sxwjfy6wpq-uc.a.run.app/chat` (production Cloud Run — see environment note)  
**Health at test time:** `healthy` — Redis, Neo4j, neo4j_graphrag, Zep, MCP all up  
**Workspace branch:** `refactor/cart-drawer-components` (10+ commits ahead of deployed `main`)

**STATUS: DONE_WITH_CONCERNS**

---

## Environment note (localhost)

**Localhost could not be exercised in this cloud-agent VM.**

| Blocker | Detail |
|---------|--------|
| Missing `.env` secrets | No Neo4j, Zep, or GCP Application Default Credentials available in the agent environment |
| Docker unavailable | `dockerd` fails on iptables/NAT (`TABLE_ADD failed: Operation not supported`) — Redis Stack (RediSearch) cannot run via compose |
| Plain Redis only | `redis-server` on :6379 responds to PING but lacks RediSearch modules required for LangGraph checkpointer |

Evaluation therefore used **gstack-browse** against the live Cloud Run deployment. Production runs `main`; the workspace branch contains post-QA fixes (rate-limit UX, budget chip fast-path, delivery-only replies, cart-drawer z-index) that are **not yet deployed**. Findings below reflect **what customers see today on production**, with branch context where relevant.

---

## 1. Executive summary

**Overall grade: B− (solid catalog and pricing fidelity; routing and support flows need polish on production)**

Tested as a real shopper through gstack-browse: birthday discovery, product detail, delivery fees, cart adds (ordinal + carousel), support FAQ, off-topic redirect, budget gifts (starter chip + natural phrasing), guest checkout, order tracking, and proceed-to-checkout. The assistant reads professionally on catalog discovery and price attribution when MCP tools succeed. Verified prices and delivery rates matched Kapruka MCP ground truth with no hallucinations observed.

**What works well**

- **Carousel discovery** after Colombo zone clarification — 9+ birthday cakes with Low Stock badges, prose recommendations tied to “elegant / not too sweet”.
- **Product detail (explicit ID)** — Springtime `CAKE00KA001685` weight **2.77 Lbs (1.25 KG)**; matches MCP exactly.
- **Delivery verification** — Colombo 05, Sunday 29 June 2026, **Rs. 300** with “verified with Kapruka” attribution; MCP confirms `available`, rate **300**.
- **Carousel add-to-cart** — Happy Birthday Symphony Ribbon Cake **Rs. 6,500** added via button; cart drawer correct (MCP: `CAKE00KA001827` → 6500).
- **Natural budget query** — “chocolate gift for my wife, budget around 5000 rupees” returned 8-item chocolate carousel (Dad Blue Heart **2,950**, Sweet Indulgence **3,230**, Fruits Harmony **4,900**, etc.) aligned with MCP search.
- **Order tracking chip** — “Track order VIMP34456CB2” returned full delivery timeline (received → out for delivery → delivered).
- **Cart drawer + composer** — `#chat-message` `is editable: true` with drawer open; fill succeeds (improvement vs prior QA).

**What needs refinement (production)**

1. **Starter chip gates discovery on Colombo sub-zone** — “Birthday cake for mom in Colombo” asks for Colombo 01–05 before showing carousel; adds friction vs prior local QA on `refactor/cart-drawer-components`.
2. **Combined product-detail + preference fails** — “less sweet + weight” returned “Please check your delivery details and try again” instead of catalog guidance.
3. **Ordinal cart resolution broken** — “add the second one” → “couldn't find a product matching 'the second one'” despite visible carousel context.
4. **Support FAQ regression** — damaged-cake refund question returned generic welcome concierge copy, not policy handoff (+94-11-7551111, policy URL).
5. **Off-topic weather misrouted** — “What's the weather in Colombo today?” triggered delivery-zone clarification instead of polite decline + gift pivot.
6. **Budget starter chip surfaces vouchers** — “Gift ideas under Rs. 5,000” carousel showed Keells/Spa/Hilton vouchers at Rs. 5,000 rather than curated chocolates/gifts under budget.
7. **Guest checkout error** — “Can I checkout as a guest…” returned “Something went wrong. Please try again.”
8. **Proceed-to-checkout misread** — cart drawer button text interpreted as city name → “couldn't find that city”.
9. **Vague delivery-fee ask** — “Colombo 05 this Sunday — delivery fee?” prompted “Which product…?” before answering (extra turn).

**Branch context:** Local browse re-QA on 2026-06-29 (same branch, localhost) reported PASS for budget chip, delivery-only replies, and rate-limit UX — fixes exist in git but are not on production `main`.

---

## 2. Test matrix

| # | Scenario | User message / action | Bot behavior | MCP validation | Result |
|---|----------|----------------------|--------------|----------------|--------|
| 1 | Birthday discovery (chip) | Clicked “Birthday cake for mom in Colombo” | Asked Colombo sub-zone (01–05) before carousel | N/A | **PARTIAL** (zone gate) |
| 2 | Discovery (follow-up) | “Colombo 05 please — elegant birthday cake, not too sweet” | 9-item cake carousel + prose picks | Springtime **5770**, Symphony **6500** | **PASS** |
| 3 | Product detail (combined) | “Tell me more about Springtime… weight and less sweet?” | “Please check your delivery details and try again.” | MCP has weight **2.77** | **FAIL** |
| 4 | Product detail (explicit) | “…CAKE00KA001685?” | “weighs 2.77 Lbs (1.25 KG)” | `kapruka_get_product` → **5770**, **2.77 lbs** | **PASS** |
| 5 | Delivery fee (vague) | “Colombo 05 this Sunday — delivery fee?” | “Which product are you interested in…?” | N/A | **PARTIAL** |
| 6 | Delivery fee (specific) | “Springtime cake… Colombo 05 on Sunday 29 June 2026” | “Rs. 300 (verified with Kapruka)” | `kapruka_check_delivery` → rate **300** | **PASS** |
| 7 | Ordinal cart | “Please add the second one to my cart” | “couldn't find a product matching 'the second one'” | Symphony **6500** in carousel | **FAIL** |
| 8 | Carousel add | Clicked carousel **Add to cart** (Symphony) | Cart: **Rs. 6,500** | `CAKE00KA001827` → **6500** | **PASS** |
| 9 | Support / FAQ | “return and refund policy if cake arrives damaged?” | Generic welcome / capability list | N/A | **FAIL** (regression) |
| 10 | Off-topic | “What's the weather in Colombo today?” | Colombo zone clarification | N/A | **FAIL** |
| 11a | Budget gift (starter) | Clicked “Gift ideas under Rs. 5,000” | Voucher carousel (Keells, Spa Ceylon, Hilton @ 5000) | Chocolates ≤5000 exist in MCP | **PARTIAL** |
| 11b | Budget gift (natural) | “chocolate gift for my wife, budget around 5000 rupees” | 8-item chocolate carousel ≤ Rs. 4,900 | Dad Blue Heart **2950**, Sweet Indulgence **3230** | **PASS** |
| 12 | Guest checkout info | “Can I checkout as a guest without creating an account?” | “Something went wrong. Please try again.” | N/A | **FAIL** |
| 13 | Proceed checkout | Cart drawer **Proceed to checkout** | “couldn't find that city… try Colombo 03, Kandy, or Galle” | N/A | **FAIL** (button text → city) |
| 14 | Order tracking | Clicked “Track order VIMP34456CB2” | Full delivery timeline through “delivered” | N/A | **PASS** |
| 15 | Cart drawer UX | Composer with drawer open | `is editable: true`; fill succeeds | N/A | **PASS** |

**MCP validation:** `.venv` + `lib/kapruka/mcp_client.MCPHttpClient` against `https://mcp.kapruka.com/mcp`.  
**MCP limits respected:** ~8 tool calls in validation script; did not invoke `kapruka_create_order` (30/hour guest limit).

---

## 3. Strengths

- **Catalog fidelity:** Springtime **Rs. 5,770** / **2.77 Lbs** and Symphony **Rs. 6,500** match MCP; no price hallucinations observed.
- **Delivery professionalism:** Parses Sunday → **29 June 2026**, canonical zone **Colombo 05**, flat fee **Rs. 300** with Kapruka attribution.
- **Natural-language budget discovery:** Wife/chocolate/5000 path curates relevant gift carousel from MCP-priced inventory.
- **Order tracking:** Starter chip with sample order ID returns structured timeline suitable for customer self-service.
- **Carousel UX:** Rich grid with images, stock badges, and working add-to-cart.
- **Infrastructure:** All five `/health` services up on production including `neo4j_graphrag`.
- **Cart drawer accessibility:** Message composer remains editable when drawer is open.

---

## 4. Gaps & refinements (prioritized)

### P0 — High customer impact

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Ordinal cart broken | “the second one” not resolved from carousel context | `graphs/nodes/resolve_cart_product.py` — restore carousel index memory |
| Support FAQ regression | Refund question → generic welcome | `lib/chat/routing.py` / support fast-path — handoff to policy URL + phone |
| Guest checkout error | “Something went wrong” on guest question | Trace checkout intent routing; add regression test |
| Proceed button → city | Drawer CTA text routed as city name | Checkout handler must ignore UI label strings |

### P1 — Important refinements

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Zone gate on birthday chip | Chip asks Colombo 01–05 before carousel | Align chip fast-path with branch `request_specificity` bypass (already on workspace branch) |
| Product detail + preference error | “less sweet” path → delivery error | `lib/chat/product_detail.py` `product_preference_note()` (on branch, not deployed) |
| Budget chip shows vouchers | Under-5000 chip ≠ chocolate curation | Budget chip should use same curation path as natural “chocolate gift 5000” |
| Off-topic weather | Weather → delivery zones | Off-topic guard in `generate_response.py` |
| Vague delivery-fee extra turn | Requires product before fee | City+date delivery-only fast-path (fix on branch) |

### P2 — Polish

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Deploy branch fixes | Production lags `refactor/cart-drawer-components` by 10+ commits | Merge + deploy to close gap with local browse re-QA |
| Localhost QA blocked in CI agents | No secrets / no Docker | Document agent prerequisites or provide read-only staging URL on feature branches |

---

## 5. MCP alignment

| Bot claim | MCP ground truth (`MCPHttpClient`) | Match? |
|-----------|--------------------------------------|--------|
| Springtime Birthday Ribbon Cake Rs. 5,770 (`CAKE00KA001685`) | `price: LKR 5770`, weight `2.77 lbs` | ✅ |
| Happy Birthday Symphony Ribbon Cake Rs. 6,500 | `CAKE00KA001827` → 6500, weight 2.44 lbs | ✅ |
| Colombo 05, 2026-06-29, Rs. 300 delivery | `available`, rate **300** | ✅ |
| Chocolate gifts ≤ Rs. 5,000 (natural query) | Dad Blue Heart **2950**, Sweet Indulgence **3230**, Fruits Harmony **4900** | ✅ |
| Budget chip vouchers @ Rs. 5,000 | MCP search `chocolate gift max_price=5000` returns chocolates, not vouchers | ⚠️ Curation mismatch |

**No price hallucinations observed** in this session.

---

## 6. UX observations

| Area | Observation |
|------|-------------|
| **Latency** | Discovery ~8–12s after zone; product detail ~12s; delivery ~12s |
| **Carousel** | Rich grid; prose recommendations reference real product names and prices |
| **Clarifying questions** | Colombo sub-zone gate on chip adds a turn; natural phrasing sometimes smoother |
| **Cart drawer** | Line items, quantity stepper, proceed button work; composer no longer blocked |
| **Checkout** | Proceed CTA and guest path broken on production |
| **Starter chips** | Tracking excellent; budget chip weak; birthday chip zone-gated |
| **Errors** | Generic “Something went wrong” on guest checkout — not actionable |

---

## 7. Recommendations

1. **Deploy `refactor/cart-drawer-components` fixes to production** — local browse re-QA (2026-06-28/29) already validated budget chip, delivery-only, rate-limit UX, and cart-drawer z-index on that branch.
2. **Restore ordinal cart resolution** — high-impact for conversational “add the second one” after carousel.
3. **Fix support FAQ and off-topic routing** — regressions vs June 27 QA on same flows.
4. **Repair checkout CTAs** — guest explanation and proceed button must not collide with city-collection step.
5. **Realign budget starter chip** with natural-language chocolate curation under Rs. 5,000.
6. **Re-run guest checkout E2E** through click-to-pay link creation (`kapruka_create_order` — respect 30/hour guest limit) after deploy.
7. **Enable localhost QA in agent environments** — provide `.env` via Secret Manager pull or feature-branch preview deploy for browse matrix runs.

---

## Appendix: tooling notes

- **Browse:** `/home/ubuntu/.cursor/skills/gstack/browse/dist/browse` (`$B`) — Playwright Chromium installed via `npx playwright install chromium`
- **MCP validation:** `.venv` + `MCPHttpClient.connect()` against `https://mcp.kapruka.com/mcp`
- **Health:** `curl -fsS https://agentic-kapruka-sxwjfy6wpq-uc.a.run.app/health`
- **Screenshot:** `/tmp/chat-qa-20260629.png` (tracking timeline)
- **MCP rate limits:** 60 req/min per IP; `kapruka_create_order` 30/hour; 30-min read cache; guest click-to-pay 60-minute locked prices
- **Localhost attempt:** `make dev` blocked — no `.env` credentials, Docker NAT/iptables failure, plain Redis without RediSearch
