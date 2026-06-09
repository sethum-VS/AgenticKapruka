# How to Set Up a Developer Environment

Complete local setup for running AgenticKapruka on your machine.

## Prerequisites

| Tool | Version | Purpose |
| --- | --- | --- |
| Python | 3.12+ | Runtime |
| Docker | Latest | Redis Stack container |
| gcloud CLI | Latest | Vertex AI ADC |
| git | Any | Clone the repository |

External accounts (free tiers available):

- [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
- [Zep Cloud](https://www.getzep.com/)
- Google Cloud project with Vertex AI enabled

## Steps

### 1. Clone and install

```bash
git clone https://github.com/sethum-VS/AgenticKapruka.git
cd AgenticKapruka
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### 2. Bootstrap environment

```bash
./scripts/bootstrap_env.sh
```

Or copy the template manually:

```bash
cp .env.example .env
```

### 3. Configure secrets

Edit `.env` with your values:

| Variable | Source |
| --- | --- |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j Aura console |
| `ZEP_API_KEY` | Zep Cloud dashboard |
| `GCP_PROJECT_ID`, `GCP_LOCATION` | Google Cloud console |
| `SESSION_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |

See [Environment reference](reference-environment.md) for all variables.

### 4. Authenticate Vertex AI

```bash
gcloud auth application-default login
```

Gemini chat uses Vertex AI by default (`GEMINI_BACKEND=vertex`). Set `GEMINI_BACKEND=api_key` and `GOOGLE_API_KEY` only for the Developer API.

### 5. Start Redis Stack

LangGraph's checkpointer requires RediSearch:

```bash
docker run -d --name agentic-kapruka-redis -p 6379:6379 redis/redis-stack-server:latest
```

### 6. Bootstrap Neo4j ontology (first time)

Run the full HybridRAG bootstrap (schema, ingest, embed, vector index):

```bash
python scripts/bootstrap_neo4j.py
```

Or run steps individually:

```bash
python scripts/migrate_ontology.py
python scripts/ingest_categories.py
python scripts/embed_ontology.py
```

`embed_ontology.py` creates the `ontology_category_embedding` vector index after embedding. `/health` reports `neo4j_graphrag: up` only when embeddings and the vector index exist.

After an embedding model upgrade, clear and re-embed:

```bash
python scripts/bootstrap_neo4j.py --skip-migrate --skip-ingest --force-reembed
```

### 7. Run the app

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000/chat](http://localhost:8000/chat).

### 8. Verify health

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

## Quality checks

Run before every commit:

```bash
ruff check .
ruff format --check .
mypy app/ lib/ graphs/
pytest tests/unit -q -m "not browser"
```

## Optional: Tailwind CSS rebuild

```bash
./scripts/install-tailwind.sh
npx tailwindcss -i static/css/input.css -o static/css/output.css --watch
```

## Optional: GPU analytics dev image

```bash
docker build -f Dockerfile.cuda -t agentic-kapruka:cuda .
docker run --gpus all agentic-kapruka:cuda
```

Production Cloud Run uses the CPU-only `Dockerfile`.

## Optional: Gunicorn (production-like)

```bash
gunicorn -c gunicorn.conf.py app.main:app
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Redis connection refused | Start Redis Stack container on port 6379 |
| Neo4j auth failed | Check Aura credentials and `bolt+s://` URI |
| Zep health check failed | Verify API key; app runs without memory if Zep is down |
| Vertex 403 | Enable Vertex AI API; confirm ADC and project ID |
| mypy errors in dev | Run `mypy app/ lib/ graphs/` only on project code |

## Related

- [Environment reference](reference-environment.md)
- [Deploy to Cloud Run](DEPLOY.md)
- [AGENTS.md](../AGENTS.md)
