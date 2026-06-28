# AgenticKapruka Local Chat QA Report

**Date:** 2026-06-27  
**Assessor:** Independent QA (gstack-browse customer dialogue + Kapruka MCP cross-validation)  
**Environment:** `http://localhost:8080/chat` (local dev, branch `refactor/cart-drawer-components`)  
**Health at test time:** `healthy` — Redis, Neo4j, neo4j_graphrag, Zep, MCP all up (brief 503 blip during Neo4j connection reset mid-session)

**STATUS: DONE_WITH_CONCERNS**

---

## 1. Executive summary

**Overall grade: A- (customer-ready for core flows, polish gaps remain)**

Tested as a real shopper: birthday discovery, product detail, delivery fees, ordinal cart adds, support FAQ, off-topic redirect, budget gifts, guest checkout guidance, carousel add-to-cart, and proceed-to-checkout. The agent reads like a professional Kapruka concierge for most paths — grounded prices, sensible boundaries, and helpful guest-checkout guidance.

**What works well**

- Carousel-first discovery for “birthday cake for mom in Colombo” (no bureaucratic date gate).
- Delivery verification with explicit date, zone, and fee attribution (“Rs. 300” for Colombo 05, Sunday 28 June 2026).
- Ordinal cart resolution (“add the second one” → Happy Birthday Symphony Ribbon Cake).
- Support FAQ handoff (phone, policy URL, clear scope limits).
- Off-topic weather redirect without being dismissive.
- Budget discovery via starter chip and natural phrasing (8 chocolate gifts under Rs. 5,000 with carousel).
- Guest checkout explanation (click-to-pay, no account).
- Carousel `POST /cart/add` returned **200** in ~2.7s this session (no new 502).
- Proceed to checkout correctly asks for delivery city (no duplicate “city not found” error observed this run).

**What still needs refinement**

1. **Product detail inconsistency** — same session said Springtime cake weight is “2.77 Lbs” in one turn, then “weight is not specified” in another.
2. **Delivery-only queries get bundled with product carousels** — fee answer arrived, but paired with redundant cake recommendations.
3. **Budget query regression in long sessions** — “wife, budget 5000” in a polluted thread returned only one odd pick (Mermaid cake set); fresh session / starter chip worked well.
4. **Presentation polish** — generic carousel blurbs (“A thoughtful Kapruka gift…”), truncated titles, ~10–20s turn latency.
5. **Cart drawer blocks chat input** — when open, message textarea is hard to target; automation and users can get stuck behind the overlay.

With weight/detail consistency fixed and carousel copy improved, this is close to production-polished for discovery → cart → guest click-to-pay.

---

## 2. Test matrix

| # | Scenario | User message / action | Bot behavior | MCP validation | Result |
|---|----------|----------------------|--------------|----------------|--------|
| 1 | Product discovery | “I need a birthday cake for my mom in Colombo” | Immediate carousel (9+ cakes), Low Stock badges, Colombo delivery note | `kapruka_get_product` Springtime **5770 LKR**; MCP bare search `birthday cake` often empty — bot uses hybrid/curation path | **PASS** |
| 2 | Product detail | “Tell me more about the Springtime Birthday Ribbon Cake…” | Returned ID `CAKE00KA001685`, **Rs. 5,770**; weight stated inconsistently across turns | MCP: **5770 LKR**, weight **2.77** | **PASS** (weight copy inconsistent) |
| 3 | Delivery fee | “Can you deliver to Colombo 05 this Sunday? What's the delivery fee?” | “Sunday, June 28, 2026… delivery fee of **Rs. 300**” (also surfaced cake list) | MCP `kapruka_check_delivery`: available=true, rate=**300**, date=2026-06-28 | **PASS** (noisy response) |
| 4 | Ordinal cart | “Please add the second one to my cart” | “Added **Happy Birthday Symphony Ribbon Cake** to your cart.” | MCP `CAKE00KA001827`: **6500 LKR**; cart drawer showed Rs. 6,500 | **PASS** |
| 5 | Support / FAQ | “What is your return and refund policy if the cake arrives damaged?” | Perishable guidance, **+94-11-7551111**, policy URL, “shopping assistant only” | N/A (policy handoff) | **PASS** |
| 6 | Off-topic | “What's the weather in Colombo today?” | Polite decline + pivot to gifts/delivery | N/A | **PASS** |
| 7a | Budget gift (starter) | Clicked “Gift ideas under Rs. 5,000” | 8-item chocolate carousel; prose cites Rs. 1,990 / 2,300 / 3,230 options | MCP: Sweet Indulgence **3230**, Dad Blue Heart **2950**, Fruits Harmony **4900** | **PASS** |
| 7b | Budget gift (natural) | “wife, budget around 5000 rupees” (long session) | Single weak pick (“Ocean Whisper Mermaid Cake Set” Rs. 3,740) | Not fully verified | **PARTIAL FAIL** (context pollution?) |
| 8 | Guest checkout info | “Can I checkout as a guest without creating an account?” | Clear click-to-pay path; mentions Proceed to checkout | N/A | **PASS** |
| 9 | Carousel add | Clicked carousel **Add to cart** | `POST /cart/add → 200` (2676ms) | N/A | **PASS** |
| 10 | Proceed checkout | Cart drawer **Proceed to checkout** | “Which Kapruka delivery city… Colombo 03, Kandy, or Galle.” | N/A | **PASS** (city step only; full guest order not completed) |

**Note:** Cursor `CallMcpTool` to `user-kapruka` timed out repeatedly; live validation succeeded via project `.venv` + `lib/kapruka/mcp_client.MCPHttpClient` against `https://mcp.kapruka.com/mcp`.

---

## 3. Strengths

- **Catalog fidelity:** Verified product prices match Kapruka MCP (`CAKE00KA001685` → 5770, `CAKE00KA001827` → 6500, chocolate gifts → 3230/2950/4900). No price hallucinations observed.
- **Delivery professionalism:** Parses “this Sunday” to **28 June 2026**, canonical zone **Colombo 05**, flat fee **Rs. 300** with MCP confirmation.
- **Ordinal cart resolution:** “The second one” maps to the second carousel item after discovery.
- **Support boundaries:** Refund FAQ deflects to Kapruka support; does not pretend to process refunds.
- **Off-topic handling:** Weather redirected without breaking rapport.
- **Budget-aware discovery:** Starter chip path is excellent — curated carousel, in-budget rationale, no interrogation loop.
- **Guest path clarity:** Guest checkout explanation aligns with Kapruka guest click-to-pay model.
- **Infrastructure:** Neo4j GraphRAG up this session — hybrid discovery paths appear active (rich birthday cake carousel despite bare MCP `birthday cake` search returning no hits).
- **Cart hardening:** Carousel add succeeded on first try; checkout proceed debounce appears improved (no spurious city error this run).

---

## 4. Gaps & refinements (prioritized)

### P0 — High customer impact

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Product attribute inconsistency | Springtime weight **2.77 Lbs** in detail turn; later “weight is not specified” | `lib/chat/product_detail.py`, `graphs/nodes/generate_response.py` — persist resolved attributes in session state |
| Long-session intent drift | “wife budget 5000” → single off-theme product after many turns | `graphs/nodes/analyze_intent.py`, context window trimming — reset discovery intent on budget/gift pivots |

### P1 — Important refinements

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Delivery-only answers bundle carousels | Delivery fee correct but followed by 3 cake recommendations | `lib/chat/intent_heuristics.py` / `is_delivery_only_inquiry` — answer fee only, skip product carousel |
| Generic carousel subtitles | Every card: “A thoughtful Kapruka gift for your occasion.” | `templates/components/product_card.html`, `lib/chat/product_curation.py` — use MCP `summary` |
| Cart drawer blocks chat input | Browse `fill` timeouts when drawer open; Escape required | `static/js/cart-drawer.js` — focus trap / close on outside click; don’t cover composer |
| Turn latency ~10–20s | Browse waits 10–18s per SSE completion | `graphs/nodes/agent_loop.py` — fast-path for FAQ, delivery, ordinal cart |

### P2 — Polish

| Issue | Evidence | Suggested fix |
|-------|----------|---------------|
| Truncated product titles | Long cake names clip in carousel headings | CSS `line-clamp` + `title` tooltip |
| Branding inconsistency | Welcome: “Kapruka AI”; header: “Kapruka Gift Assistant” | `templates/chat/index.html` |
| Historical 502 on cart add | Console retained earlier `502 from /cart/add` | Monitor in prod; keep `app/routes/cart.py` retry + user-visible error partial |
| Neo4j connection blips | Brief `/health` 503 + `ConnectionResetError` in logs | Connection pooling / retry in Neo4j client |

---

## 5. MCP alignment

| Bot claim | MCP ground truth (`MCPHttpClient`) | Match? |
|-----------|--------------------------------------|--------|
| Springtime Birthday Ribbon Cake Rs. 5,770 (`CAKE00KA001685`) | `price.amount: 5770`, weight `2.77` | ✅ |
| Happy Birthday Symphony Ribbon Cake Rs. 6,500 | `CAKE00KA001827` → 6500, weight 2.44 | ✅ |
| Colombo 05, 2026-06-28, Rs. 300 delivery | `available: true, rate: 300` | ✅ |
| Sweet Indulgence Chocolate Gift Box Rs. 3,230 | search `chocolate gift box` → 3230 | ✅ |
| Dad Blue Heart Chocolate Gift Box Rs. 2,950 | search → 2950 | ✅ |
| MCP bare search `birthday cake` | Often “No products found” | ⚠️ App hybrid path still surfaces real cakes — not hallucination, but MCP search ≠ in-app ranking |

**No price hallucinations observed** in this session.

---

## 6. UX observations

| Area | Observation |
|------|-------------|
| **Latency** | SSE turns ~10–20s; send button disables during generation (good); still slow for mobile |
| **Carousel** | Rich grid with images, stock badges, Add to cart; generic blurbs reduce trust |
| **Clarifying questions** | Discovery no longer date-gated first (improved); delivery queries could be leaner |
| **Cart drawer** | Shows line items, quantity stepper, proceed button; blocks composer when open |
| **Checkout** | Proceed correctly enters `delivery_city` step; guest click-to-pay not exercised end-to-end |
| **Starter chips** | High quality — “Gift ideas under Rs. 5,000” is an excellent onboarding path |
| **Guest mode** | “Kapruka Guest” label + LKR currency selector clear |
| **Errors** | No new 502 this session; older console errors from prior runs still visible |

---

## 7. Recommendations

1. **Fix product detail memory** — once weight/ID fetched via MCP, don’t contradict in follow-up turns (`product_detail.py`).
2. **Lean delivery-only replies** — fee + date + availability only; skip carousel unless user asks for products.
3. **Improve carousel copy** — replace generic blurb with MCP `summary` snippets.
4. **Cart drawer UX** — auto-close or don’t obscure message input after Proceed; keep `proceedCheckoutInFlight` debounce.
5. **Add regression tests** for: delivery-only (no carousel), product detail consistency, budget gift in fresh vs. long sessions.
6. **Re-run guest checkout E2E** through click-to-pay link creation (`kapruka_create_order` — respect 30/hour guest limit).
7. **Monitor Neo4j stability** — brief health degradation during cloud connection resets.

---

## Appendix: tooling notes

- **Browse:** `~/.cursor/skills/gstack/browse/dist/browse` (`$B`)
- **MCP validation:** `.venv` + `MCPHttpClient.connect()` (Cursor `CallMcpTool` timed out)
- **Health:** `curl -fsS http://localhost:8080/health`
- **Logs:** `.dev/backend.log` — checkout intent routing, Neo4j connection errors
- **MCP rate limits:** 60 req/min per IP; `kapruka_create_order` 30/hour; 30-min read cache
