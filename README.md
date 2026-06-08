# AgenticKapruka

Agentic shopping assistant for [Kapruka](https://www.kapruka.com): FastAPI + HTMX + LangGraph + Neo4j GraphRAG, deployed to Google Cloud Run.

## Stack

- **API / UI**: FastAPI, Jinja2, HTMX, Alpine.js, Tailwind CSS
- **Agent**: LangGraph orchestration, Google GenAI / Vertex
- **Data**: Neo4j GraphRAG, Redis (sessions/cart), Zep (memory)
- **Integration**: Kapruka MCP server for live product and order data

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Bootstrap a local `.env` from gcloud (or copy `.env.example` and fill values manually):

```bash
./scripts/bootstrap_env.sh
# Edit NEO4J_* and ZEP_API_KEY in .env, then:
uvicorn app.main:app --reload
```

## Graph analytics backends

Community detection for co-purchase recommendations runs on **NetworkX** (CPU) in production Cloud Run (`lib/analytics/networkx_worker.py`). That path is the default everywhere and requires no GPU.

An optional **cuGraph** GPU path exists for local development only (`lib/analytics/cugraph_optional.py`). It imports cuGraph lazily when CUDA is available and falls back to NetworkX otherwise. Cloud Run deploys do **not** use the cuGraph image.

Build the optional CUDA dev image (requires NVIDIA Container Toolkit and a GPU):

```bash
docker build -f Dockerfile.cuda -t agentic-kapruka:cuda .
docker run --gpus all agentic-kapruka:cuda
```

Build and run the production image (slim CPU runtime for Cloud Run):

```bash
docker build -t agentic-kapruka .
docker run --rm -p 8080:8080 --env-file .env agentic-kapruka
curl -s http://localhost:8080/health
# {"status":"healthy","services":{"redis":{"status":"up"},"neo4j":{"status":"up"},"zep":{"status":"up"},"mcp":{"status":"up"}}}
```

The container starts Gunicorn with Uvicorn workers via `gunicorn.conf.py` (`workers = 2 * cpu_count + 1`, `timeout = 120` for SSE streams, `graceful_timeout` / `keepalive` tuned for Cloud Run). Local equivalent:

```bash
gunicorn -c gunicorn.conf.py app.main:app
```

## Development

Run quality checks before every commit:

```bash
ruff check .
ruff format --check .
mypy app/ lib/ graphs/
pytest tests/unit -q
```

See [AGENTS.md](AGENTS.md) for agent and contributor conventions.

## Cloud Run deployment

Production deploy steps (Artifact Registry, VPC connector, Secret Manager, `gcloud run deploy`):

```bash
./scripts/deploy_cloud_run.sh --dry-run   # preview commands
./scripts/deploy_cloud_run.sh              # build, push, deploy
```

GitHub Actions CI/CD (secrets + `main` branch deploy workflow):

```bash
GCP_SA_KEY_FILE=./sa-key.json ./scripts/setup_github_cicd.sh
```

Full walkthrough: [docs/DEPLOY.md](docs/DEPLOY.md).

## Ralph autonomous workflow

PRD backlog lives in `prd.json`. Progress is logged in `progress.txt`.

```bash
# Single supervised iteration (headless JSON stream — good for logs)
./scripts/ralph-once.sh

# Interactive Cursor Agent UI (full-screen TUI)
./scripts/ralph-once.sh --interactive
# or: RALPH_INTERACTIVE=1 ./scripts/ralph-once.sh

# Headless plain text instead of JSON
./scripts/ralph-once.sh --text

# AFK loop (default 10 iterations, headless JSON stream)
./scripts/ralph.sh

# AFK loop with readable plain text instead of JSON
./scripts/ralph.sh 10 --text

# Custom iteration count and model
./scripts/ralph.sh 5 --model composer-2.5
```

Work happens on branch `ralph/sprint-1`. One PRD item per commit: `feat(PRD-XXX): title`.

## Project layout

```
app/          FastAPI routes, config, templating
lib/          Business logic and service clients
graphs/       LangGraph agent definitions
templates/    Jinja2 HTML partials
static/       Compiled CSS and assets
tests/        Unit and integration tests
scripts/      Ralph loop and tooling
.cursor/      Cursor agent skills (Ralph personas)
```
