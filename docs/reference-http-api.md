# HTTP API Reference

HTTP routes exposed by the FastAPI application. All HTML responses use Jinja2 templates; chat streaming uses Server-Sent Events (SSE).

Base URL: `http://localhost:8000` (local) or your Cloud Run service URL (production).

## Health

### `GET /health`

Aggregate dependency probe.

**Response 200** (all services up):

```json
{
  "status": "healthy",
  "services": {
    "redis": {"status": "up"},
    "neo4j": {"status": "up"},
    "neo4j_graphrag": {"status": "up"},
    "zep": {"status": "up"},
    "mcp": {"status": "up"}
  }
}
```

`neo4j_graphrag` is `up` only when Category embeddings and the `ontology_category_embedding` vector index exist (run `python scripts/bootstrap_neo4j.py` against Aura first).

**Response 503** (any service down): `"status": "degraded"`

## Chat

### `GET /chat`

Renders the main chat page (HTML).

### `POST /chat/stream`

Streams assistant response as SSE events.

| Parameter | Type | Location | Required |
| --- | --- | --- | --- |
| `message` | string | form body | Yes |

**Response:** `text/event-stream` with HTML fragments per event.

Requires session cookie (`SESSION_COOKIE_NAME`).

## Cart

All cart routes return HTML partials for HTMX swap.

### `POST /cart/add`

Add item to session cart.

### `POST /cart/remove`

Remove item from cart.

### `POST /cart/update`

Update item quantity.

## Checkout

### `GET /checkout`

Checkout page shell (HTML).

### `POST /checkout/check-delivery`

Validate delivery city and date. Returns HTML status partial.

### `POST /checkout/validate-delivery`

Validate delivery form fields.

### `POST /checkout/validate-recipient`

Validate recipient name and phone.

### `POST /checkout/validate-sender`

Validate sender details and gift message.

## Session

### `POST /session/currency`

Set session currency preference.

| Value | Code |
| --- | --- |
| Sri Lankan Rupee | `LKR` |
| US Dollar | `USD` |
| British Pound | `GBP` |
| Australian Dollar | `AUD` |
| Canadian Dollar | `CAD` |
| Euro | `EUR` |

## Partials (HTMX fragments)

### `GET /partials/search`

Product search results partial.

### `GET /partials/delivery-cities`

Delivery city autocomplete suggestions.

### `GET /partials`

Other HTMX partial endpoints as registered in `app/routes/partials.py`.

## Static assets

### `GET /static/*`

Compiled CSS, JavaScript, and images from `static/`.

## Root redirect

### `GET /`

Redirects to `/chat` (HTTP 307).

## Security

- No CORS middleware — same-origin requests only
- Session cookie signed with `SESSION_SECRET` (minimum 32 characters)
- No authentication layer — session-scoped anonymous access

## Related

- [Customer capabilities](reference-customer-capabilities.md)
- [Environment reference](reference-environment.md)
