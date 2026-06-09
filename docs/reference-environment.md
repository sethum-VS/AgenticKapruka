# Environment Variables Reference

Complete configuration reference for AgenticKapruka. Copy `.env.example` to `.env` for local development. Production uses Google Secret Manager mapped at deploy time.

## Required variables

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `REDIS_URL` | URL | `redis://localhost:6379/0` | Redis Stack connection. Production: `rediss://` via Memorystore VPC |
| `NEO4J_URI` | URI | — | Neo4j AuraDB bolt URI (`bolt+s://...`) |
| `NEO4J_USER` | string | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | secret | — | Neo4j password |
| `ZEP_API_KEY` | secret | — | Zep Cloud API key |
| `SESSION_SECRET` | secret | — | Cookie signing key, minimum 32 characters |
| `GCP_PROJECT_ID` | string | — | Google Cloud project for Vertex AI (Gemini chat + `gemini-embedding-2`) |
| `GCP_LOCATION` | string | `us-central1` | Vertex AI region for Gemini chat (embeddings use the global endpoint) |

## Google AI

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_BACKEND` | enum | `vertex` | `vertex` (ADC) or `api_key` (Developer API) |
| `GOOGLE_API_KEY` | secret | — | Required only when `GEMINI_BACKEND=api_key` |

Local Vertex setup:

```bash
gcloud auth application-default login
```

Cloud Run: attach a service account with `roles/aiplatform.user`.

## Kapruka integration

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `KAPRUKA_MCP_URL` | URL | `https://mcp.kapruka.com/mcp` | Kapruka MCP JSON-RPC endpoint |

## Production-only notes

### Redis (Memorystore)

- Reachable only from VPC — Cloud Run needs a Serverless VPC Access connector
- Use `rediss://` when TLS/AUTH is enabled

```bash
gcloud compute networks vpc-access connectors create agentic-kapruka-connector \
  --region=REGION --network=default --range=10.8.0.0/28
```

### Secret Manager mapping

Deploy with `--set-secrets`:

```
REDIS_URL=redis-url:latest
NEO4J_PASSWORD=neo4j-password:latest
ZEP_API_KEY=zep-api-key:latest
SESSION_SECRET=session-secret:latest
```

See [DEPLOY.md](DEPLOY.md) for the full deploy script.

### Neo4j GraphRAG bootstrap

HybridRAG requires a one-time Aura bootstrap before production `/health` reports healthy:

```bash
python scripts/bootstrap_neo4j.py
```

After an embedding model upgrade, re-embed with `python scripts/bootstrap_neo4j.py --skip-migrate --skip-ingest --force-reembed`.

Check GCP and GitHub prerequisites before first deploy:

```bash
./scripts/verify_production_prerequisites.sh
```

## Generating secrets

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Related

- [Developer setup](howto-developer-setup.md)
- [.env.example](../.env.example)
