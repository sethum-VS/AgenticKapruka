#!/usr/bin/env bash
# Infrastructure-as-code setup for GitHub Actions CI/CD → Google Cloud Run.
#
# Uses the GitHub CLI (gh) to configure repository secrets and scaffolds
# .github/workflows/main.yml with quality gates plus deploy on push to main.
#
# Prerequisites:
#   - gh auth login (repo secret write access)
#   - gcloud CLI (read project/region defaults; optional SA key creation)
#   - GCP service account JSON with roles: run.admin, artifactregistry.writer,
#     cloudbuild.builds.editor, secretmanager.secretAccessor, iam.serviceAccountUser
#
# Usage:
#   ./scripts/setup_github_cicd.sh
#   ./scripts/setup_github_cicd.sh --dry-run
#   GCP_SA_KEY_FILE=./sa-key.json ./scripts/setup_github_cicd.sh --secrets-only
#   ./scripts/setup_github_cicd.sh --workflow-only --force
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_FILE="${GITHUB_WORKFLOW_FILE:-${ROOT_DIR}/.github/workflows/main.yml}"
DEFAULT_REGION="us-central1"
DRY_RUN=false
FORCE=false
SECRETS_ONLY=false
WORKFLOW_ONLY=false

SERVICE_NAME="${SERVICE_NAME:-agentic-kapruka}"
AR_REPO="${AR_REPO:-agentic-kapruka}"
VPC_CONNECTOR="${VPC_CONNECTOR:-agentic-kapruka-connector}"

usage() {
  cat <<'EOF'
Usage: setup_github_cicd.sh [OPTIONS]

Configure GitHub repository secrets for Cloud Run deploy and scaffold
.github/workflows/main.yml (quality gates + deploy on push to main).

Prerequisites:
  gh auth login          GitHub CLI authenticated with secret write access
  gcloud auth login      GCP CLI for project/region defaults (optional)

Environment:
  GCP_SA_KEY_FILE        Path to GCP service account JSON key (required for secrets)
  GCP_PROJECT_ID         GCP project (default: gcloud config project)
  GCP_REGION             Deploy region (default: gcloud compute/region or us-central1)
  GITHUB_WORKFLOW_FILE   Override workflow output path (for tests)
  SERVICE_NAME           Cloud Run service name (default: agentic-kapruka)
  AR_REPO                Artifact Registry repo (default: agentic-kapruka)
  VPC_CONNECTOR          VPC Access connector name (default: agentic-kapruka-connector)

Options:
  --dry-run, -n       Print planned gh/gcloud actions without applying
  --force, -f         Overwrite existing main.yml workflow
  --secrets-only      Configure gh secrets only (skip workflow scaffold)
  --workflow-only     Scaffold workflow only (skip gh secrets)
  --help, -h          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run | -n)
      DRY_RUN=true
      shift
      ;;
    --force | -f)
      FORCE=true
      shift
      ;;
    --secrets-only)
      SECRETS_ONLY=true
      shift
      ;;
    --workflow-only)
      WORKFLOW_ONLY=true
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

if [[ "${SECRETS_ONLY}" == "true" && "${WORKFLOW_ONLY}" == "true" ]]; then
  echo "Error: --secrets-only and --workflow-only are mutually exclusive." >&2
  exit 1
fi

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

resolve_gcp_config() {
  if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
    if command -v gcloud >/dev/null 2>&1; then
      GCP_PROJECT_ID="$(gcloud_config_value project || true)"
    fi
  fi
  if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
    echo "Error: GCP_PROJECT_ID is not set and gcloud project is unset." >&2
    echo "Run: gcloud config set project PROJECT_ID" >&2
    exit 1
  fi

  if [[ -z "${GCP_REGION:-}" ]]; then
    if command -v gcloud >/dev/null 2>&1 && GCP_REGION="$(gcloud_config_value compute/region)"; then
      :
    else
      GCP_REGION="${DEFAULT_REGION}"
      echo "Note: GCP_REGION unset; using ${GCP_REGION}" >&2
    fi
  fi
}

ensure_gh_auth() {
  require_cmd gh
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "# gh auth status (skipped in dry run)"
    return 0
  fi
  if ! gh auth status >/dev/null 2>&1; then
    echo "Error: gh is not authenticated. Run: gh auth login" >&2
    exit 1
  fi
}

set_github_secrets() {
  resolve_gcp_config

  local sa_key_file="${GCP_SA_KEY_FILE:-}"
  if [[ -z "${sa_key_file}" ]]; then
    echo "Error: GCP_SA_KEY_FILE is required to set GCP_SA_KEY repository secret." >&2
    echo "Export GCP_SA_KEY_FILE=/path/to/service-account.json" >&2
    exit 1
  fi
  if [[ ! -f "${sa_key_file}" ]]; then
    echo "Error: service account key file not found: ${sa_key_file}" >&2
    exit 1
  fi

  echo "Configuring GitHub repository secrets for ${GCP_PROJECT_ID} (${GCP_REGION})" >&2

  run_cmd gh secret set GCP_SA_KEY <"${sa_key_file}"
  run_cmd gh secret set GCP_PROJECT_ID --body "${GCP_PROJECT_ID}"
  run_cmd gh secret set GCP_REGION --body "${GCP_REGION}"
  run_cmd gh secret set GCP_SERVICE_NAME --body "${SERVICE_NAME}"
  run_cmd gh secret set GCP_AR_REPO --body "${AR_REPO}"
  run_cmd gh secret set GCP_VPC_CONNECTOR --body "${VPC_CONNECTOR}"

  echo "GitHub secrets configured: GCP_SA_KEY, GCP_PROJECT_ID, GCP_REGION," >&2
  echo "  GCP_SERVICE_NAME, GCP_AR_REPO, GCP_VPC_CONNECTOR" >&2
}

write_workflow_file() {
  local workflow_dir
  workflow_dir="$(dirname "${WORKFLOW_FILE}")"

  if [[ -f "${WORKFLOW_FILE}" && "${FORCE}" != "true" ]]; then
    echo "Error: ${WORKFLOW_FILE} already exists. Use --force to overwrite." >&2
    exit 1
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "# Would write workflow: ${WORKFLOW_FILE}"
    return 0
  fi

  mkdir -p "${workflow_dir}"

  cat >"${WORKFLOW_FILE}" <<'WORKFLOW_EOF'
name: Main

on:
  push:
    branches: [main]

concurrency:
  group: main-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-and-test:
    name: Ruff, mypy, and unit tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Ruff check
        run: ruff check .

      - name: Ruff format
        run: ruff format --check .

      - name: Mypy
        run: mypy app/ lib/ graphs/

      - name: Unit tests
        run: pytest tests/unit -q

  e2e-smoke:
    name: Playwright E2E smoke tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Install Playwright Chromium
        run: playwright install --with-deps chromium

      - name: E2E smoke tests
        run: pytest tests/e2e -q

  ragas-eval:
    name: Ragas golden dataset eval
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run Ragas evaluation (mock MCP)
        run: python -m evals.ragas_eval --ci

  deploy:
    name: Deploy to Cloud Run
    needs: [lint-and-test, e2e-smoke, ragas-eval]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    env:
      GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
      GCP_REGION: ${{ secrets.GCP_REGION }}
      SERVICE_NAME: ${{ secrets.GCP_SERVICE_NAME }}
      AR_REPO: ${{ secrets.GCP_AR_REPO }}
      VPC_CONNECTOR: ${{ secrets.GCP_VPC_CONNECTOR }}
      KAPRUKA_MCP_URL: https://mcp.kapruka.com/mcp
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Enable required APIs
        run: |
          gcloud services enable \
            artifactregistry.googleapis.com \
            cloudbuild.googleapis.com \
            run.googleapis.com \
            secretmanager.googleapis.com \
            vpcaccess.googleapis.com \
            --project="${GCP_PROJECT_ID}"

      - name: Ensure Artifact Registry repository
        run: |
          if ! gcloud artifacts repositories describe "${AR_REPO}" \
            --project="${GCP_PROJECT_ID}" \
            --location="${GCP_REGION}" >/dev/null 2>&1; then
            gcloud artifacts repositories create "${AR_REPO}" \
              --project="${GCP_PROJECT_ID}" \
              --location="${GCP_REGION}" \
              --repository-format=docker \
              --description="AgenticKapruka container images"
          fi

      - name: Build and push image
        run: |
          IMAGE_TAG="${GITHUB_SHA::7}"
          IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:${IMAGE_TAG}"
          echo "IMAGE_URI=${IMAGE_URI}" >> "${GITHUB_ENV}"
          gcloud builds submit --project="${GCP_PROJECT_ID}" --tag "${IMAGE_URI}" .

      - name: Deploy to Cloud Run
        run: |
          SECRETS="REDIS_URL=redis-url:latest"
          SECRETS+=",NEO4J_URI=neo4j-uri:latest"
          SECRETS+=",NEO4J_USER=neo4j-user:latest"
          SECRETS+=",NEO4J_PASSWORD=neo4j-password:latest"
          SECRETS+=",ZEP_API_KEY=zep-api-key:latest"
          SECRETS+=",GOOGLE_API_KEY=google-api-key:latest"
          SECRETS+=",SESSION_SECRET=session-secret:latest"
          ENV_VARS="GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_LOCATION=${GCP_REGION},KAPRUKA_MCP_URL=${KAPRUKA_MCP_URL}"

          gcloud run deploy "${SERVICE_NAME}" \
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
            --vpc-connector="${VPC_CONNECTOR}" \
            --vpc-egress=private-ranges-only \
            --set-secrets="${SECRETS}" \
            --set-env-vars="${ENV_VARS}"

      - name: Verify deployment health
        run: |
          SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
            --project="${GCP_PROJECT_ID}" \
            --region="${GCP_REGION}" \
            --format='value(status.url)')"
          echo "Service URL: ${SERVICE_URL}"
          curl -fsS "${SERVICE_URL}/health"
WORKFLOW_EOF

  echo "Wrote ${WORKFLOW_FILE}" >&2
}

verify_workflow_yaml() {
  if [[ ! -f "${WORKFLOW_FILE}" ]]; then
    return 0
  fi

  python3 - "${WORKFLOW_FILE}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text(encoding="utf-8")

if "\t" in content:
    raise SystemExit("workflow contains tab characters")

required = ("on:", "jobs:", "lint-and-test:", "deploy:", "gcloud run deploy", "google-github-actions/auth")
missing = [token for token in required if token not in content]
if missing:
    raise SystemExit(f"workflow missing required tokens: {missing}")

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
  # Structural checks above are sufficient when PyYAML is unavailable.
    sys.exit(0)

data = yaml.safe_load(content)
if not isinstance(data, dict):
    raise SystemExit("workflow root must be a mapping")
if "jobs" not in data or "deploy" not in data["jobs"]:
    raise SystemExit("workflow must define deploy job")
PY
}

main() {
  if [[ "${WORKFLOW_ONLY}" != "true" ]]; then
    ensure_gh_auth
    set_github_secrets
  fi

  if [[ "${SECRETS_ONLY}" != "true" ]]; then
    write_workflow_file
    if [[ "${DRY_RUN}" != "true" ]]; then
      verify_workflow_yaml
    fi
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete — no changes applied."
  else
    echo "GitHub CI/CD setup complete."
    echo "  Workflow: ${WORKFLOW_FILE}"
    echo "  Triggers: push to main (quality gates → Cloud Run deploy)"
    echo "  Ensure GCP Secret Manager secrets exist (see docs/DEPLOY.md)."
  fi
}

main "$@"
