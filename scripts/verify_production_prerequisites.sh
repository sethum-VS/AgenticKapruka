#!/usr/bin/env bash
# Read-only checklist: verify GCP + GitHub prerequisites before first production deploy.
#
# Usage:
#   export GCP_PROJECT_ID=your-project
#   export GCP_REGION=us-central1
#   ./scripts/verify_production_prerequisites.sh
set -euo pipefail

GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
GCP_REGION="${GCP_REGION:-$(gcloud config get-value compute/region 2>/dev/null || echo us-central1)}"
SERVICE_NAME="${SERVICE_NAME:-agentic-kapruka}"
AR_REPO="${AR_REPO:-agentic-kapruka}"
VPC_CONNECTOR="${VPC_CONNECTOR:-agentic-kapruka-connector}"
RUN_SA="${RUN_SERVICE_ACCOUNT:-vertexai-api}"

if [[ -z "${GCP_PROJECT_ID}" || "${GCP_PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: set GCP_PROJECT_ID or gcloud config project" >&2
  exit 1
fi

PASS=0
FAIL=0
WARN=0

check() {
  local label="$1"
  local result="$2"
  if [[ "${result}" == "ok" ]]; then
    echo "  [OK]   ${label}"
    PASS=$((PASS + 1))
  elif [[ "${result}" == "warn" ]]; then
    echo "  [WARN] ${label}"
    WARN=$((WARN + 1))
  else
    echo "  [FAIL] ${label}"
    FAIL=$((FAIL + 1))
  fi
}

echo "Production prerequisites for ${GCP_PROJECT_ID} (${GCP_REGION})"
echo

echo "GCP APIs"
for api in aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  run.googleapis.com secretmanager.googleapis.com vpcaccess.googleapis.com; do
  if gcloud services list --enabled --project="${GCP_PROJECT_ID}" --filter="name:${api}" \
    --format='value(name)' 2>/dev/null | grep -q "${api}"; then
    check "${api}" ok
  else
    check "${api} (enable with gcloud services enable)" fail
  fi
done

echo
echo "Artifact Registry"
if gcloud artifacts repositories describe "${AR_REPO}" \
  --project="${GCP_PROJECT_ID}" --location="${GCP_REGION}" >/dev/null 2>&1; then
  check "repository ${AR_REPO}" ok
else
  check "repository ${AR_REPO}" fail
fi

echo
echo "VPC connector"
if gcloud compute networks vpc-access connectors describe "${VPC_CONNECTOR}" \
  --project="${GCP_PROJECT_ID}" --region="${GCP_REGION}" >/dev/null 2>&1; then
  check "connector ${VPC_CONNECTOR}" ok
else
  check "connector ${VPC_CONNECTOR}" fail
fi

echo
echo "Cloud Run service"
if gcloud run services describe "${SERVICE_NAME}" \
  --project="${GCP_PROJECT_ID}" --region="${GCP_REGION}" >/dev/null 2>&1; then
  check "service ${SERVICE_NAME} (exists)" warn
else
  check "service ${SERVICE_NAME} (not deployed yet)" ok
fi

echo
echo "Runtime service account"
RUN_SA_EMAIL="${RUN_SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
if gcloud iam service-accounts describe "${RUN_SA_EMAIL}" \
  --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
  check "${RUN_SA_EMAIL}" ok
else
  check "${RUN_SA_EMAIL}" fail
fi

echo
echo "Secret Manager"
for secret in redis-url neo4j-uri neo4j-user neo4j-password zep-api-key session-secret; do
  if gcloud secrets describe "${secret}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    check "secret ${secret}" ok
  else
    check "secret ${secret}" fail
  fi
done

echo
echo "GitHub repository secrets"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  for secret in GCP_SA_KEY GCP_PROJECT_ID GCP_REGION GCP_SERVICE_NAME GCP_AR_REPO GCP_VPC_CONNECTOR; do
    if gh secret list 2>/dev/null | awk '{print $1}' | grep -qx "${secret}"; then
      check "GitHub secret ${secret}" ok
    else
      check "GitHub secret ${secret}" fail
    fi
  done
  if gh secret list 2>/dev/null | awk '{print $1}' | grep -qx "GCP_RUN_SERVICE_ACCOUNT"; then
    check "GitHub secret GCP_RUN_SERVICE_ACCOUNT" ok
  else
    check "GitHub secret GCP_RUN_SERVICE_ACCOUNT (optional, defaults to vertexai-api)" warn
  fi
else
  check "gh CLI authenticated (skip GitHub secret checks)" warn
fi

echo
echo "Summary: ${PASS} ok, ${WARN} warn, ${FAIL} fail"
if [[ "${FAIL}" -gt 0 ]]; then
  echo "Fix failures before first deploy. See docs/DEPLOY.md" >&2
  exit 1
fi
echo "Ready for Neo4j bootstrap and deploy when Aura + Memorystore are configured."
