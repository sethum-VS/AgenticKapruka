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

## CI/CD

Automated GitHub Actions deployment on push to `main` is covered in PRD-086 (`scripts/setup_github_cicd.sh`).
