# Deploying AgenticKapruka to Google Cloud Run

Step-by-step guide for building the production container image, pushing to Artifact Registry, and deploying to Cloud Run with a VPC connector for Memorystore Redis and Secret Manager for credentials.

For local development env bootstrap, see `./scripts/bootstrap_env.sh` and [README.md](../README.md).

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Billing enabled on your GCP project
- Permissions: `roles/run.admin`, `roles/artifactregistry.admin`, `roles/cloudbuild.builds.editor`, `roles/secretmanager.admin`, `roles/vpcaccess.admin`
- Neo4j AuraDB and Zep Cloud accounts (external services, public internet)
- Memorystore for Redis instance on a VPC subnet reachable via Serverless VPC Access

Set your default project and region:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
```

## 1. Enable APIs

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  vpcaccess.googleapis.com
```

## 2. Artifact Registry — create repository and push image

Create a Docker repository (once per project/region):

```bash
export GCP_PROJECT_ID="$(gcloud config get-value project)"
export GCP_REGION="$(gcloud config get-value compute/region)"

gcloud artifacts repositories create agentic-kapruka \
  --project="${GCP_PROJECT_ID}" \
  --location="${GCP_REGION}" \
  --repository-format=docker \
  --description="AgenticKapruka container images"
```

### Option A — Cloud Build (recommended)

Build from the repo root and push in one step:

```bash
export IMAGE_TAG="$(git rev-parse --short HEAD)"
export IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/agentic-kapruka/agentic-kapruka:${IMAGE_TAG}"

gcloud builds submit --project="${GCP_PROJECT_ID}" --tag "${IMAGE_URI}" .
```

### Option B — Local Docker build and push

```bash
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
```

## 3. Memorystore Redis and VPC connector

Cloud Run must reach Memorystore over a private IP. Create a Serverless VPC Access connector on the same VPC as your Redis instance:

```bash
gcloud compute networks vpc-access connectors create agentic-kapruka-connector \
  --project="${GCP_PROJECT_ID}" \
  --region="${GCP_REGION}" \
  --network=default \
  --range=10.8.0.0/28
```

Use `rediss://` in `REDIS_URL` when Memorystore AUTH/TLS is enabled. Neo4j AuraDB and Zep Cloud are reached over the public internet — only Redis traffic needs the VPC connector (`--vpc-egress=private-ranges-only`).

## 4. Secret Manager — store credentials

Create secrets matching the names expected by `scripts/deploy_cloud_run.sh` (override with `SECRET_*` env vars if you use different names):

```bash
# Example: create from your local .env values (never commit these commands with real secrets)
echo -n 'rediss://10.0.0.3:6378/0' | gcloud secrets create redis-url --data-file=-
echo -n 'bolt+s://xxxx.databases.neo4j.io' | gcloud secrets create neo4j-uri --data-file=-
echo -n 'neo4j' | gcloud secrets create neo4j-user --data-file=-
echo -n 'your-password' | gcloud secrets create neo4j-password --data-file=-
echo -n 'zep_xxx' | gcloud secrets create zep-api-key --data-file=-
echo -n 'AIza...' | gcloud secrets create google-api-key --data-file=-
python3 -c "import secrets; print(secrets.token_urlsafe(48), end='')" | gcloud secrets create session-secret --data-file=-
```

Grant the Cloud Run service account access to each secret:

```bash
export PROJECT_NUMBER="$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')"
export RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for secret in redis-url neo4j-uri neo4j-user neo4j-password zep-api-key google-api-key session-secret; do
  gcloud secrets add-iam-policy-binding "${secret}" \
    --member="serviceAccount:${RUN_SA}" \
    --role="roles/secretmanager.secretAccessor"
done
```

## 5. Environment variable checklist

| Variable | Source | Required | Notes |
|----------|--------|----------|-------|
| `REDIS_URL` | Secret Manager | Yes | Memorystore private IP via VPC connector |
| `NEO4J_URI` | Secret Manager | Yes | AuraDB `bolt+s://` URI |
| `NEO4J_USER` | Secret Manager | Yes | Usually `neo4j` |
| `NEO4J_PASSWORD` | Secret Manager | Yes | AuraDB password |
| `ZEP_API_KEY` | Secret Manager | Yes | Zep Cloud API key |
| `GOOGLE_API_KEY` | Secret Manager | Yes | Gemini via Generative Language API |
| `SESSION_SECRET` | Secret Manager | Yes | ≥32 random characters |
| `GCP_PROJECT_ID` | `--set-env-vars` | Yes | Vertex text-embedding project |
| `GCP_LOCATION` | `--set-env-vars` | Yes | Vertex region (e.g. `us-central1`) |
| `KAPRUKA_MCP_URL` | `--set-env-vars` | No | Defaults to `https://mcp.kapruka.com/mcp` |

## 6. Deploy to Cloud Run

Manual deploy with `gcloud run deploy`:

```bash
gcloud run deploy agentic-kapruka \
  --project="${GCP_PROJECT_ID}" \
  --region="${GCP_REGION}" \
  --platform=managed \
  --image="${IMAGE_URI}" \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=10 \
  --memory=1Gi \
  --cpu=1 \
  --concurrency=80 \
  --timeout=120 \
  --port=8080 \
  --vpc-connector=agentic-kapruka-connector \
  --vpc-egress=private-ranges-only \
  --set-secrets="REDIS_URL=redis-url:latest,NEO4J_URI=neo4j-uri:latest,NEO4J_USER=neo4j-user:latest,NEO4J_PASSWORD=neo4j-password:latest,ZEP_API_KEY=zep-api-key:latest,GOOGLE_API_KEY=google-api-key:latest,SESSION_SECRET=session-secret:latest" \
  --set-env-vars="GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_LOCATION=${GCP_REGION},KAPRUKA_MCP_URL=https://mcp.kapruka.com/mcp"
```

`--min-instances=0` keeps cost at zero when idle; increase for lower cold-start latency.

### Automated deploy script

After secrets and the VPC connector exist:

```bash
./scripts/deploy_cloud_run.sh
```

Preview commands without making changes:

```bash
./scripts/deploy_cloud_run.sh --dry-run
```

Redeploy an existing image tag without rebuilding:

```bash
IMAGE_TAG=abc1234 ./scripts/deploy_cloud_run.sh --skip-build
```

## 7. Verify deployment

```bash
export SERVICE_URL="$(gcloud run services describe agentic-kapruka \
  --region="${GCP_REGION}" \
  --format='value(status.url)')"

curl -s "${SERVICE_URL}/health" | jq .
# {"status":"healthy","services":{"redis":{"status":"up"},...}}
```

A `degraded` response means one or more backends (Redis, Neo4j, Zep, MCP) is unreachable — check Secret Manager values and VPC connector routing for Redis.

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Redis `down` in `/health` | VPC connector missing/wrong region, or `REDIS_URL` points to unreachable IP |
| Neo4j `down` | Incorrect Aura URI or credentials in secrets |
| Zep `down` | Invalid `ZEP_API_KEY` |
| MCP `down` | Egress blocked or wrong `KAPRUKA_MCP_URL` |
| Container fails to start | Missing secret or IAM `secretAccessor` on Run service account |

## CI/CD — GitHub Actions → Cloud Run

Automated deployment on push to `main` is configured via `scripts/setup_github_cicd.sh`. The script uses the GitHub CLI to set repository secrets and scaffolds `.github/workflows/main.yml` with quality gates (Ruff, mypy, unit tests, Playwright E2E, Ragas eval) followed by a Cloud Run deploy job.

### Prerequisites

| Tool | Purpose |
|------|---------|
| [GitHub CLI (`gh`)](https://cli.github.com/) | `gh auth login` with permission to set repository secrets |
| [gcloud CLI](https://cloud.google.com/sdk/docs/install) | Read `GCP_PROJECT_ID` / region defaults; create deploy service account |
| GCP service account JSON key | CI deploy identity with `roles/run.admin`, `roles/artifactregistry.writer`, `roles/cloudbuild.builds.editor`, `roles/secretmanager.secretAccessor`, `roles/iam.serviceAccountUser` |

Create a deploy service account (example):

```bash
export GCP_PROJECT_ID="$(gcloud config get-value project)"
export DEPLOY_SA="github-actions-deploy@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create github-actions-deploy \
  --display-name="GitHub Actions Cloud Run deploy"

for role in run.admin artifactregistry.writer cloudbuild.builds.editor \
  secretmanager.secretAccessor iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${DEPLOY_SA}" \
    --role="roles/${role}"
done

gcloud iam service-accounts keys create sa-key.json \
  --iam-account="${DEPLOY_SA}"
```

Ensure GCP Secret Manager secrets and the VPC connector exist (sections 3–4 above) before the first deploy.

### One-time setup

```bash
export GCP_SA_KEY_FILE=./sa-key.json
./scripts/setup_github_cicd.sh
```

This configures these GitHub repository secrets:

| Secret | Value |
|--------|-------|
| `GCP_SA_KEY` | Service account JSON key (full file contents) |
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Deploy region (e.g. `us-central1`) |
| `GCP_SERVICE_NAME` | Cloud Run service name (default `agentic-kapruka`) |
| `GCP_AR_REPO` | Artifact Registry repository (default `agentic-kapruka`) |
| `GCP_VPC_CONNECTOR` | VPC Access connector name (default `agentic-kapruka-connector`) |

Options:

```bash
./scripts/setup_github_cicd.sh --dry-run          # preview gh secret set + workflow path
./scripts/setup_github_cicd.sh --secrets-only       # secrets only, skip workflow write
./scripts/setup_github_cicd.sh --workflow-only -f   # regenerate main.yml
```

### Workflow behavior

On every push to `main`, `.github/workflows/main.yml`:

1. Runs lint, type-check, unit tests, Playwright E2E smoke tests, and Ragas eval (mock MCP).
2. When all jobs pass, builds the production image with `gcloud builds submit`, pushes to Artifact Registry, and runs `gcloud run deploy` with Secret Manager env vars (same mapping as `scripts/deploy_cloud_run.sh`).
3. Verifies deployment with `curl` against `/health`.

Pull requests continue to use `.github/workflows/ci.yml` for faster feedback without deploy.
