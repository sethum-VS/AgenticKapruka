"""Tests for scripts/deploy_cloud_run.sh and docs/DEPLOY.md."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy_cloud_run.sh"
DEPLOY_DOC = REPO_ROOT / "docs" / "DEPLOY.md"


def test_deploy_script_exists_and_is_executable() -> None:
    assert DEPLOY_SCRIPT.is_file()
    mode = DEPLOY_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "deploy_cloud_run.sh should be executable"


def test_deploy_script_contract() -> None:
    """Deploy script covers Artifact Registry, image push, and Cloud Run deploy."""
    content = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert "artifacts repositories create" in content
    assert "gcloud builds submit" in content
    assert "docker build" in content
    assert "docker push" in content
    assert "gcloud run deploy" in content
    assert "--vpc-connector" in content
    assert "--vpc-egress=private-ranges-only" in content
    assert "--set-secrets" in content
    assert "--min-instances" in content
    assert "REDIS_URL=" in content
    assert "NEO4J_URI=" in content
    assert "ZEP_API_KEY=" in content
    assert "GOOGLE_API_KEY=" in content
    assert "GCP_PROJECT_ID=" in content


def test_deploy_doc_contract() -> None:
    """DEPLOY.md documents registry, push, VPC connector, secrets, and env checklist."""
    content = DEPLOY_DOC.read_text(encoding="utf-8")

    assert "artifacts repositories create" in content
    assert "gcloud builds submit" in content
    assert "docker push" in content
    assert "gcloud run deploy" in content
    assert "--vpc-connector" in content
    assert "--set-secrets" in content
    assert "--min-instances=0" in content
    assert "REDIS_URL" in content
    assert "NEO4J_URI" in content
    assert "ZEP_API_KEY" in content
    assert "GOOGLE_API_KEY" in content
    assert "GCP_PROJECT_ID" in content
    assert "vpc-access connectors create" in content


def test_deploy_dry_run_prints_gcloud_commands() -> None:
    """Dry run emits planned gcloud invocations without requiring live GCP."""
    env = os.environ.copy()
    env["GCP_PROJECT_ID"] = "mock-gcp-project"
    env["GCP_REGION"] = "us-central1"
    env["IMAGE_TAG"] = "testtag"

    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--dry-run", "--skip-build"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert "mock-gcp-project" in output
    assert "us-central1" in output
    assert "gcloud run deploy" in output
    assert "--vpc-connector" in output
    assert "--set-secrets" in output
    assert "REDIS_URL=redis-url:latest" in output
    assert "NEO4J_URI=neo4j-uri:latest" in output
    assert "ZEP_API_KEY=zep-api-key:latest" in output
    assert "GOOGLE_API_KEY=google-api-key:latest" in output
    assert "GCP_PROJECT_ID=mock-gcp-project" in output
    assert "Dry run complete" in output
