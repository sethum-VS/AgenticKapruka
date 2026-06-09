#!/usr/bin/env bash
# Build, push, and deploy AgenticKapruka to Google Cloud Run.
#
# Prerequisites:
#   - gcloud CLI authenticated with deploy permissions
#   - Secret Manager secrets created (see docs/DEPLOY.md)
#   - Serverless VPC Access connector for Memorystore Redis
#
# Usage:
#   ./scripts/deploy_cloud_run.sh              # build, push, deploy
#   ./scripts/deploy_cloud_run.sh --dry-run    # print planned gcloud commands
#   ./scripts/deploy_cloud_run.sh --skip-build # deploy existing image tag only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

SERVICE_NAME="${SERVICE_NAME:-agentic-kapruka}"
AR_REPO="${AR_REPO:-agentic-kapruka}"
VPC_CONNECTOR="${VPC_CONNECTOR:-agentic-kapruka-connector}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MEMORY="${MEMORY:-1Gi}"
CPU="${CPU:-1}"
CONCURRENCY="${CONCURRENCY:-80}"
TIMEOUT="${TIMEOUT:-120}"
KAPRUKA_MCP_URL="${KAPRUKA_MCP_URL:-https://mcp.kapruka.com/mcp}"
DEFAULT_REGION="us-central1"
DRY_RUN=false
SKIP_BUILD=false

# Secret Manager secret names (values mapped to env vars via --set-secrets).
SECRET_REDIS_URL="${SECRET_REDIS_URL:-redis-url}"
SECRET_NEO4J_URI="${SECRET_NEO4J_URI:-neo4j-uri}"
SECRET_NEO4J_USER="${SECRET_NEO4J_USER:-neo4j-user}"
SECRET_NEO4J_PASSWORD="${SECRET_NEO4J_PASSWORD:-neo4j-password}"
SECRET_ZEP_API_KEY="${SECRET_ZEP_API_KEY:-zep-api-key}"
SECRET_SESSION_SECRET="${SECRET_SESSION_SECRET:-session-secret}"

usage() {
  cat <<'EOF'
Usage: deploy_cloud_run.sh [--dry-run] [--skip-build]

Build the production Docker image, push to Artifact Registry, and deploy to Cloud Run
with a VPC connector (Memorystore Redis) and Secret Manager env vars.

Environment overrides:
  SERVICE_NAME          Cloud Run service name (default: agentic-kapruka)
  AR_REPO               Artifact Registry repository (default: agentic-kapruka)
  VPC_CONNECTOR         VPC Access connector name (default: agentic-kapruka-connector)
  IMAGE_TAG             Image tag (default: short git SHA or "latest")
  GCP_PROJECT_ID        GCP project (default: gcloud config project)
  GCP_REGION            Deploy region (default: gcloud compute/region or us-central1)
  MIN_INSTANCES         Cloud Run min instances (default: 0)
  KAPRUKA_MCP_URL       Non-secret MCP endpoint (default: https://mcp.kapruka.com/mcp)

Options:
  --dry-run, -n   Print gcloud commands without executing
  --skip-build    Deploy an existing image tag (skip gcloud builds submit)
  --help, -h      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run | -n)
      DRY_RUN=true
      shift
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

gcloud_config_value() {
  local value
  value="$(gcloud config get-value "$1" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -z "${value}" || "${value}" == "(unset)" ]]; then
    return 1
  fi
  printf '%s' "${value}"
}

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

ensure_prereqs() {
  require_cmd gcloud
  if [[ "${SKIP_BUILD}" != "true" ]]; then
    require_cmd git
  fi
}

resolve_config() {
  if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
    GCP_PROJECT_ID="$(gcloud_config_value project || true)"
  fi
  if [[ -z "${GCP_PROJECT_ID}" ]]; then
    echo "Error: GCP_PROJECT_ID is not set and gcloud project is unset." >&2
    echo "Run: gcloud config set project PROJECT_ID" >&2
    exit 1
  fi

  if [[ -z "${GCP_REGION:-}" ]]; then
    if GCP_REGION="$(gcloud_config_value compute/region)"; then
      :
    else
      GCP_REGION="${DEFAULT_REGION}"
      echo "Note: gcloud compute/region is unset; using ${GCP_REGION}" >&2
    fi
  fi

  if [[ -z "${RUN_SERVICE_ACCOUNT:-}" ]]; then
    RUN_SERVICE_ACCOUNT="vertexai-api@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
  fi

  if [[ -z "${IMAGE_TAG:-}" ]]; then
    if git -C "${ROOT_DIR}" rev-parse --short HEAD >/dev/null 2>&1; then
      IMAGE_TAG="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
    else
      IMAGE_TAG="latest"
    fi
  fi

  IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:${IMAGE_TAG}"
}

artifact_registry_create_cmd() {
  cat <<EOF
gcloud artifacts repositories create ${AR_REPO} \\
  --project=${GCP_PROJECT_ID} \\
  --location=${GCP_REGION} \\
  --repository-format=docker \\
  --description="AgenticKapruka container images"
EOF
}

docker_push_steps() {
  cat <<EOF
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
EOF
}

cloud_build_submit_cmd() {
  printf 'gcloud builds submit --project=%s --tag %s .\n' "${GCP_PROJECT_ID}" "${IMAGE_URI}"
}

run_deploy_secrets() {
  local secrets
  secrets="REDIS_URL=${SECRET_REDIS_URL}:latest"
  secrets+=",NEO4J_URI=${SECRET_NEO4J_URI}:latest"
  secrets+=",NEO4J_USER=${SECRET_NEO4J_USER}:latest"
  secrets+=",NEO4J_PASSWORD=${SECRET_NEO4J_PASSWORD}:latest"
  secrets+=",ZEP_API_KEY=${SECRET_ZEP_API_KEY}:latest"
  secrets+=",SESSION_SECRET=${SECRET_SESSION_SECRET}:latest"
  printf '%s' "${secrets}"
}

cloud_run_deploy_cmd() {
  local secrets env_vars
  secrets="$(run_deploy_secrets)"
  env_vars="GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_LOCATION=${GCP_REGION},GEMINI_BACKEND=vertex,KAPRUKA_MCP_URL=${KAPRUKA_MCP_URL}"

  cat <<EOF
gcloud run deploy ${SERVICE_NAME} \\
  --project=${GCP_PROJECT_ID} \\
  --region=${GCP_REGION} \\
  --platform=managed \\
  --image=${IMAGE_URI} \\
  --allow-unauthenticated \\
  --min-instances=${MIN_INSTANCES} \\
  --max-instances=${MAX_INSTANCES} \\
  --memory=${MEMORY} \\
  --cpu=${CPU} \\
  --concurrency=${CONCURRENCY} \\
  --timeout=${TIMEOUT} \\
  --port=8080 \\
  --vpc-connector=${VPC_CONNECTOR} \\
  --vpc-egress=private-ranges-only \\
  --service-account=${RUN_SERVICE_ACCOUNT} \\
  --set-secrets=${secrets} \\
  --set-env-vars=${env_vars}
EOF
}

ensure_apis() {
  local apis=(
    artifactregistry.googleapis.com
    cloudbuild.googleapis.com
    run.googleapis.com
    secretmanager.googleapis.com
    vpcaccess.googleapis.com
  )
  for api in "${apis[@]}"; do
    run_cmd gcloud services enable "${api}" --project="${GCP_PROJECT_ID}"
  done
}

ensure_artifact_registry() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "# Create Artifact Registry repository (no-op if it already exists):"
    artifact_registry_create_cmd
    return 0
  fi

  if gcloud artifacts repositories describe "${AR_REPO}" \
    --project="${GCP_PROJECT_ID}" \
    --location="${GCP_REGION}" >/dev/null 2>&1; then
    echo "Artifact Registry repository exists: ${AR_REPO}"
    return 0
  fi

  echo "Creating Artifact Registry repository: ${AR_REPO}" >&2
  run_cmd gcloud artifacts repositories create "${AR_REPO}" \
    --project="${GCP_PROJECT_ID}" \
    --location="${GCP_REGION}" \
    --repository-format=docker \
    --description="AgenticKapruka container images"
}

build_and_push() {
  if [[ "${SKIP_BUILD}" == "true" ]]; then
    echo "Skipping image build; deploying tag ${IMAGE_URI}" >&2
    return 0
  fi

  echo "Building and pushing ${IMAGE_URI}" >&2
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "# Option A — Cloud Build (recommended):"
    cloud_build_submit_cmd
    echo "# Option B — local Docker build and push:"
    docker_push_steps
    return 0
  fi

  run_cmd gcloud builds submit --project="${GCP_PROJECT_ID}" --tag "${IMAGE_URI}" .
}

deploy_service() {
  echo "Deploying Cloud Run service ${SERVICE_NAME} in ${GCP_REGION}" >&2
  if [[ "${DRY_RUN}" == "true" ]]; then
    cloud_run_deploy_cmd
    return 0
  fi

  local secrets env_vars
  secrets="$(run_deploy_secrets)"
  env_vars="GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_LOCATION=${GCP_REGION},GEMINI_BACKEND=vertex,KAPRUKA_MCP_URL=${KAPRUKA_MCP_URL}"

  run_cmd gcloud run deploy "${SERVICE_NAME}" \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_REGION}" \
    --platform=managed \
    --image="${IMAGE_URI}" \
    --allow-unauthenticated \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --memory="${MEMORY}" \
    --cpu="${CPU}" \
    --concurrency="${CONCURRENCY}" \
    --timeout="${TIMEOUT}" \
    --port=8080 \
    --vpc-connector="${VPC_CONNECTOR}" \
    --vpc-egress=private-ranges-only \
    --service-account="${RUN_SERVICE_ACCOUNT}" \
    --set-secrets="${secrets}" \
    --set-env-vars="${env_vars}"
}

print_env_checklist() {
  cat <<'EOF'

Required runtime configuration (Secret Manager → env var):
  REDIS_URL        Memorystore private IP (rediss:// when AUTH enabled)
  NEO4J_URI        Neo4j AuraDB bolt URI
  NEO4J_USER       Neo4j username (usually neo4j)
  NEO4J_PASSWORD   Neo4j password
  ZEP_API_KEY      Zep Cloud API key
  SESSION_SECRET   Cookie signing secret (≥32 chars)

Non-secret env vars set on deploy:
  GEMINI_BACKEND   vertex (Gemini via Vertex AI + service account ADC)
  GCP_PROJECT_ID   Vertex AI project
  GCP_LOCATION     Vertex region (e.g. us-central1)
  RUN_SERVICE_ACCOUNT  Cloud Run runtime SA (default: vertexai-api@PROJECT.iam.gserviceaccount.com)
  KAPRUKA_MCP_URL  Kapruka MCP endpoint (default public URL)

See docs/DEPLOY.md for full setup steps.
EOF
}

main() {
  ensure_prereqs
  resolve_config

  echo "Project:  ${GCP_PROJECT_ID}"
  echo "Region:   ${GCP_REGION}"
  echo "Image:    ${IMAGE_URI}"
  echo "Service:  ${SERVICE_NAME}"
  echo "VPC:      ${VPC_CONNECTOR}"
  echo "Min inst: ${MIN_INSTANCES}"

  ensure_apis
  ensure_artifact_registry
  build_and_push
  deploy_service
  print_env_checklist

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete — no changes applied."
  else
    echo "Deploy complete. Verify: curl \$(gcloud run services describe ${SERVICE_NAME} --region=${GCP_REGION} --format='value(status.url)')/health"
  fi
}

main "$@"
