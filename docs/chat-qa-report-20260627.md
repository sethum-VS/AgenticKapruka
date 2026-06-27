# AgenticKapruka Local Chat QA Report

**Date:** 2026-06-27  
**Assessor:** Independent QA (gstack-browse + Kapruka MCP cross-validation)  
**Environment:** `http://localhost:8080/chat` (local dev)  
**Health:** `degraded` — Redis/Zep/MCP up; Neo4j + neo4j_graphrag **down**

**STATUS: DONE_WITH_CONCERNS**

---

## Post-fix verification

**Date:** 2026-06-27 (post parallel-agent fixes)  
**Verifier:** Integration verifier (unit tests + SSE matrix + gstack-browse spot-checks)  
**Environment:** `http://127.0.0.1:8080/chat` — health **healthy** (Redis, Neo4j, neo4j_graphrag, Zep, MCP all up)  
**Fix files landed:** `app/routes/cart.py`, `static/js/cart-drawer.js`, `lib/chat/request_specificity.py`, `templates/chat/index.html`

### Unit / integration tests

| Suite | Result |
|-------|--------|
| `test_agent_loop`, `test_request_specificity`, `test_resolve_cart_product` + related new files | **187 passed** |
| `tests/integration/test_cart_flow.py` | **5 passed** (HTMX add returns 200 + visible error on upstream failure, not silent 502) |

### Browse matrix (scenarios 1–8)

| # | Scenario | Result | Notes |
|---|----------|--------|-------|
| 1 | Birthday cake Colombo — carousel before date nag | **PASS** | Carousel in first response (~9.5s); no date-only gate blocking browse |
| 2 | Springtime cake details | **PASS** | Rs. 5,770 / product detail in reply (~7.6s) |
| 3 | Colombo 05 Sunday delivery fee | **FIXED** (unit) | `is_delivery_only_inquiry` + early route to `resolve_delivery_context` in `retrieve_hybrid_context.py`; browse re-check pending |
| 4 | “Add the second one” | **PASS** | Ordinal cart confirmation (~2.9s) |
| 5 | Damaged cake refund FAQ | **PASS** | Support handoff with policy/refund copy (~0.7s) |
| 6 | Weather off-topic | **PASS** | Polite redirect (~2.3s) |
| 7 | Wife budget 5000 | **PASS** | Curated carousel under budget (~7.9s) |
| 7c | Proceed checkout — NO double city error | **PASS** | First “Proceed to checkout” asks city; duplicate fire did **not** produce “couldn't find that city” (`cart-drawer.js` `proceedCheckoutInFlight` debounce) |
| 8 | Carousel Add to cart — NO 502 | **PASS** | Three `POST /cart/add` observed at **200** (2.1–4.7s); curl HTMX add also **200**; no new 502 in this session |

### Remaining concerns

1. **Delivery-only queries (scenario 3)** — fixed via `is_delivery_only_inquiry` bypassing product specificity gate; browse re-check recommended.
2. **Cart add latency** — still 2–5s per carousel click (MCP `get_product`); acceptable but not polished.
3. **Dev stability during parallel edits** — uvicorn `--reload` restarts from concurrent agent saves caused intermittent `Connection refused` / SSE `Failed to fetch` during browse; verification used a no-reload backend for the matrix run.
4. **P2 polish unchanged** — carousel title truncation, generic card blurb, ~15–30s turn latency.

### STATUS

**DONE_WITH_CONCERNS** — P0/P1 targets (carousel-first discovery, checkout debounce, cart-add 502 hardening) verified green; **scenario 3 delivery fee** is a new regression to triage before customer-ready sign-off.

---

## 1. Executive summary

**Overall grade: B+ (promising, not yet production-polished)**

The chat agent behaves like a knowledgeable Kapruka shopping concierge for most core flows: gift discovery, product detail, delivery verification, ordinal cart references, support FAQ handoff, and off-topic redirection all felt professional and grounded in real catalog data. Prices and product IDs cited in conversation matched Kapruka MCP on every spot-check.

Key concerns blocking a confident “ready for real customers” rating:

1. **Infrastructure dependency** — local Neo4j GraphRAG is down; hybrid context is degraded to MCP-only search paths.
2. **Carousel “Add to cart” reliability** — one observed `502` on `POST /cart/add` (console: `Response Status Error Code 502`); retries succeeded but took ~4.7s.
3. **Checkout UX friction** — “Proceed to checkout” can fire twice (drawer button injects the phrase into chat), producing a confusing “city not found” error before the user supplies a city.
4. **Presentation polish** — truncated product titles in carousel/cards and generic card copy reduce trust.

With Neo4j bootstrapped, carousel add hardened, and checkout step-gating tightened, this agent is close to customer-ready for discovery → cart → guest click-to-pay.

---

## 2. Test matrix

| # | Scenario | User message | Bot response quality | MCP validation | Result |
|---|----------|--------------|----------------------|----------------|--------|
| 1 | Product discovery | “I need a birthday cake for my mom in Colombo” | Asked for delivery date (`2026-06-27` minimum) before/alongside carousel; eventually showed 9 birthday cakes with prices and Low Stock badges | MCP `kapruka_search_products(q='birthday cake')` returned **no hits** (bot likely used hybrid/curated path); carousel names/prices align with MCP product lookups | **PASS** (concerns: date gate before browse) |
| 2 | Specific product | “Tell me more about the Springtime Birthday Ribbon Cake…” | Returned `CAKE00KA001685`, weight 2.77 lbs, description snippet, **Rs. 5,770** | `kapruka_get_product`: name match, price **5770 LKR** | **PASS** |
| 3 | Delivery | “Can you deliver to Colombo 05 this Sunday? What's the delivery fee?” | “Delivery to Colombo 05 on Sunday, 28 June 2026: **Rs. 300** (verified with Kapruka)” | `kapruka_check_delivery`: available=true, rate=300 | **PASS** |
| 4 | Cart / ordinal | “Please add the second one to my cart” | “Added **Happy Birthday Symphony Ribbon Cake** to your cart.” Cart showed 1× @ Rs. 6,500 | `kapruka_get_product(CAKE00KA001827)`: **6500 LKR** | **PASS** |
| 5 | Support / FAQ | “What is your return and refund policy if the cake arrives damaged?” | Perishable guidance, support phone **+94-11-7551111**, policy URL, clear scope boundary (“I'm a shopping assistant only”) | N/A (policy handoff, not catalog) | **PASS** |
| 6 | Off-topic redirect | “What's the weather in Colombo today?” | Polite decline + pivot to gifts/delivery | N/A | **PASS** |
| 7a | Vague gift / budget | “I want something nice for my wife, budget around 5000 rupees” | Curated 5 cakes under Rs. 5,000 with rationale; no excessive clarifying loop | Crimson cake **3800 LKR**, Teddy **3900 LKR** via MCP | **PASS** |
| 7b | Guest checkout info | “Can I checkout as a guest without creating an account?” | Clear guest click-to-pay instructions; mentions Proceed to checkout | N/A | **PASS** |
| 7c | Guest checkout flow | Clicked **Proceed to checkout** (cart drawer) | Asked delivery city; then erroneous “couldn't find that city” when phrase re-fired | N/A | **PARTIAL FAIL** |
| 8 | Carousel UI add | Clicked carousel **Add to cart** | First attempt: **502**; retry: **200** in ~4.7s, item added | N/A | **PASS** (flaky) |

**Screenshot:** `/tmp/chat-eval-checkout.png` — guest checkout Q&A + checkout city error visible.

---

## 3. Strengths

- **Catalog fidelity:** Product IDs, names, and LKR prices matched Kapruka MCP on all verified items (Springtime, Symphony, Crimson/Teddy bento cakes).
- **Delivery professionalism:** Explicit date parsing (“Sunday, 28 June 2026”), canonical city (“Colombo 05”), fee with “verified with Kapruka” attribution.
- **Ordinal cart resolution:** “The second one” correctly resolved to the second carousel item (Happy Birthday Symphony Ribbon Cake).
- **Support boundaries:** Refund FAQ appropriately deflects to Kapruka support with phone + policy link; does not hallucinate refund processing.
- **Off-topic handling:** Weather query redirected without being dismissive.
- **Budget-aware discovery:** Wife / Rs. 5,000 request produced relevant, in-budget cake carousel without a frustrating 3-question interrogation.
- **Guest path clarity:** Guest checkout explanation mentions click-to-pay and no login — aligns with Kapruka guest commerce model.
- **No console JS errors** during normal chat turns (one network 502 on cart add).

---

## 4. Gaps & refinements (prioritized)

### P0 — Blockers / high customer impact

| Issue | Evidence | Suggested fix area |
|-------|----------|-------------------|
| Neo4j GraphRAG down locally | `/health` → `neo4j` + `neo4j_graphrag` down | Run `python scripts/bootstrap_neo4j.py`; verify in deploy health gate |
| Intermittent carousel cart failure | Console: `502 from /cart/add` at 11:03 UTC; succeeded on retry | `app/routes/cart.py`, `lib/kapruka` MCP client retries; surface user-visible retry/toast |

### P1 — Important refinements

| Issue | Evidence | Suggested fix area |
|-------|----------|-------------------|
| Checkout double-fire / city error | After Proceed click: “Which city…?” then “I couldn't find that city…” without user input | `static/js/cart-drawer.js` (injects “Proceed to checkout”), `graphs/nodes/analyze_intent.py`, `graphs/nodes/run_checkout_graph` — debounce or ignore duplicate proceed while `delivery_city` step active |
| Discovery asks date before showing options (text-only) | First reply: “I have not verified Kapruka delivery… When would you like delivery?” | `lib/chat/request_specificity.py`, `graphs/nodes/agent_loop.py` — show carousel first for situational gifts; collect date at checkout |
| Truncated product titles | “Crimson Love Gold Chocolate Sponge Bento Cake With Chocolate **Hea**” in carousel and bot prose | `templates/chat/` carousel partial, CSS `line-clamp` / title field |
| Slow cart add (~4–5s) | Network: `POST /cart/add → 200 (4670ms)` | `app/routes/cart.py` `get_product` on add — cache or parallelize |

### P2 — Polish

| Issue | Evidence | Suggested fix area |
|-------|----------|-------------------|
| Generic carousel subtitle | Every card: “A thoughtful Kapruka gift for your occasion.” | `templates/chat/message_assistant.html` / curation copy in `lib/chat/product_curation.py` |
| Turn latency ~15–30s | Browse waits 15–30s per response | `graphs/nodes/agent_loop.py` planner iterations; consider Flash fast-path |
| Branding inconsistency | Header: “Kapruka Gift Assistant”; footer: “Kapruka Concierge can make mistakes” | `templates/chat/index.html` |
| Vague “Colombo” vs zone | User said “Colombo”; bot asked date, not zone — later “Colombo 05” worked | `graphs/nodes/resolve_delivery_context.py` — gentle zone prompt |

---

## 5. MCP alignment

| Bot claim | MCP ground truth | Match? |
|-----------|------------------|--------|
| Springtime Birthday Ribbon Cake Rs. 5,770 (`CAKE00KA001685`) | `price.amount: 5770` | ✅ |
| Happy Birthday Symphony Ribbon Cake Rs. 6,500 | `CAKE00KA001827` → 6500 | ✅ |
| Crimson Love bento Rs. 3,800 | `CAKE00KA002079` → 3800 | ✅ |
| Colombo 05, 2026-06-28, Rs. 300 delivery | `available: true, rate: 300` | ✅ |
| MCP search `birthday cake` (standalone) | “No products found” | ⚠️ Bot still surfaced cakes via app search/hybrid path — not a hallucination, but MCP bare search differs from in-app behavior |

**No price hallucinations observed.** Product descriptions were abbreviated but factually consistent with MCP `description` fields.

---

## 6. UX observations

| Area | Observation |
|------|-------------|
| **Latency** | SSE stream completes in ~15–30s per turn; send button disabled during generation (good); feels slow for mobile shoppers |
| **Carousel** | Rich product grid with images, Low Stock badges, Add to cart + View links; titles truncate awkwardly |
| **Clarifying questions** | Delivery-date prompt on first cake query may feel bureaucratic; budget gift query skipped unnecessary questions (good) |
| **Cart drawer** | Chat-based add reliable; UI button flaky once; badge updates after drawer open |
| **Checkout** | Proceed button correctly routes to `run_checkout_graph`; city collection works; duplicate trigger confusing |
| **Error handling** | 502 on cart add silent to user (HTMX only); checkout city error message helpful but mistimed |
| **Starter chips** | Useful suggestions (birthday cake, roses, budget gifts, track order) |
| **Guest mode** | “Kapruka Guest” label clear; currency selector (LKR default) present |

---

## 7. Recommendations

1. **Bootstrap Neo4j before QA/eval** (`python scripts/bootstrap_neo4j.py`) — restores hybrid ranking and birthday-cake curation per `AGENTS.md`.
2. **Harden `/cart/add`** — retry MCP `get_product` on 5xx; return HTMX error partial with “Try again” instead of silent 502 (`app/routes/cart.py`).
3. **Debounce Proceed to checkout** — prevent `cart-drawer.js` from submitting while a checkout turn is in-flight (`static/js/cart-drawer.js`, checkout graph state).
4. **Discovery fast-path** — for situational queries with city (“cake for mom in Colombo”), show curated carousel immediately; defer date to checkout unless user asks about timing (`lib/chat/request_specificity.py`, `agent_loop.py`).
5. **Fix carousel title truncation** — use full `name` with tooltip or two-line wrap (`templates/chat/`).
6. **Replace generic card blurb** with occasion-aware or product-summary snippet from MCP `summary`.
7. **Add regression tests** for: carousel add (integration), proceed-checkout debounce, ordinal cart in multi-carousel session.
8. **Re-run this matrix** after Neo4j up with `make dev` healthy — expect improved discovery ranking.

---

## Appendix: tooling notes

- **Browse:** `$B` at `~/.cursor/skills/gstack/browse/dist/browse`
- **MCP via Cursor CallMcpTool:** timed out; validation succeeded via project `.venv` + `lib/kapruka/mcp_client.py`
- **Logs:** `.dev/backend.log` confirms `analyze_intent: proceed-to-checkout trigger from cart drawer` → `run_checkout_graph` with `checkout_state: delivery_city`
