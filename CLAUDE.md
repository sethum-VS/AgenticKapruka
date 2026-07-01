# AgenticKapruka — Claude / gstack context

Agentic shopping assistant for Kapruka: FastAPI + HTMX + LangGraph + Neo4j GraphRAG on Google Cloud Run.

See [AGENTS.md](AGENTS.md) for coding conventions and quality bar.

## Deploy Configuration (configured by /setup-deploy)

- Platform: Google Cloud Run (GitHub Actions CD)
- Production URL: assigned on first deploy — resolve with `gcloud run services describe agentic-kapruka --region=us-central1 --format='value(status.url)'`
- Deploy workflow: `.github/workflows/main.yml` (quality gates → `gcloud builds submit` → `gcloud run deploy` on push to `main`)
- Deploy status command: `gh run list --branch main --limit 3 --json workflowName,status,conclusion`
- Merge method: squash (repo default)
- Project type: web app (FastAPI + HTMX)
- Post-deploy health check: `{SERVICE_URL}/health` — expects `{"status":"healthy",...}` with all five services up (including `neo4j_graphrag`)

### GCP resources

| Resource | Value |
|----------|-------|
| Project | `project-3bb9c91c-69ed-4507-998` |
| Region | `us-central1` |
| Service | `agentic-kapruka` |
| Artifact Registry | `agentic-kapruka` |
| VPC connector | `agentic-kapruka-connector` (`10.9.0.0/28`) |
| Memorystore Redis | `agentic-kapruka-redis` |
| Runtime SA | `vertexai-api@project-3bb9c91c-69ed-4507-998.iam.gserviceaccount.com` |
| Deploy SA | `github-actions-deployer@project-3bb9c91c-69ed-4507-998.iam.gserviceaccount.com` |

### Custom deploy hooks

- Pre-merge: CI must pass (Ruff, mypy, unit tests, Playwright E2E, Ragas eval)
- Deploy trigger: automatic on push to `main` after CI jobs succeed
- Deploy status: poll GitHub Actions `Main` workflow deploy job; then `curl -fsS "${SERVICE_URL}/health"`
- Health check: `/health` — run `python scripts/bootstrap_neo4j.py` before first deploy or local eval if `neo4j_graphrag` is down; also run locally before evals when hybrid context returns zero products
- Prerequisites: `./scripts/verify_production_prerequisites.sh`
