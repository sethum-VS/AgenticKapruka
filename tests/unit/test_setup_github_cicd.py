"""Tests for scripts/setup_github_cicd.sh and scaffolded main.yml workflow."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_github_cicd.sh"
MAIN_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "main.yml"
DEPLOY_DOC = REPO_ROOT / "docs" / "DEPLOY.md"


def test_setup_script_exists_and_is_executable() -> None:
    assert SETUP_SCRIPT.is_file()
    mode = SETUP_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "setup_github_cicd.sh should be executable"


def test_setup_script_contract() -> None:
    """Setup script configures gh secrets and scaffolds main.yml."""
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "gh secret set GCP_SA_KEY" in content
    assert "gh secret set GCP_PROJECT_ID" in content
    assert "gh secret set GCP_REGION" in content
    assert "gh auth status" in content
    assert ".github/workflows/main.yml" in content
    assert "gcloud run deploy" in content
    assert "gcloud builds submit" in content
    assert "google-github-actions/auth" in content
    assert "verify_workflow_yaml" in content


def test_main_workflow_contract() -> None:
    """Scaffolded main.yml runs quality gates and deploys to Cloud Run on main."""
    content = MAIN_WORKFLOW.read_text(encoding="utf-8")

    assert "branches: [main]" in content
    assert "lint-and-test:" in content
    assert "e2e-smoke:" in content
    assert "ragas-eval:" in content
    assert "deploy:" in content
    assert "needs: [lint-and-test, e2e-smoke, ragas-eval]" in content
    assert "ruff check" in content
    assert "mypy app/ lib/ graphs/" in content
    assert "pytest tests/unit" in content
    assert "pytest tests/e2e" in content
    assert "evals.ragas_eval --ci" in content
    assert "google-github-actions/auth@v2" in content
    assert "google-github-actions/setup-gcloud@v2" in content
    assert "gcloud builds submit" in content
    assert "gcloud run deploy" in content
    assert "--vpc-connector" in content
    assert "--set-secrets" in content
    assert "curl -fsS" in content


def test_main_workflow_yaml_syntax() -> None:
    """Workflow file is valid YAML when PyYAML is available."""
    content = MAIN_WORKFLOW.read_text(encoding="utf-8")
    assert "\t" not in content

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return

    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    assert "jobs" in data
    assert "deploy" in data["jobs"]
    deploy_needs = data["jobs"]["deploy"].get("needs", [])
    assert "lint-and-test" in deploy_needs
    assert "e2e-smoke" in deploy_needs
    assert "ragas-eval" in deploy_needs


def _write_fake_gh(fake_bin: Path) -> None:
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "auth" && "${2:-}" == "status" ]]; then
  exit 0
fi

if [[ "${1:-}" == "secret" && "${2:-}" == "set" ]]; then
  name="${3:-}"
  if [[ "${name}" == "GCP_SA_KEY" ]]; then
  # stdin key payload
    cat >/dev/null
    exit 0
  fi
  if [[ "${4:-}" == "--body" ]]; then
    exit 0
  fi
fi

echo "fake gh: unexpected invocation: $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_setup_dry_run_secrets_and_workflow(tmp_path: Path) -> None:
    """Dry run prints gh secret set and workflow path without live GitHub."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_gh(fake_bin)

    sa_key = tmp_path / "sa-key.json"
    sa_key.write_text('{"type":"service_account"}', encoding="utf-8")

    workflow_out = tmp_path / "main.yml"

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GCP_PROJECT_ID"] = "mock-gcp-project"
    env["GCP_REGION"] = "us-central1"
    env["GCP_SA_KEY_FILE"] = str(sa_key)
    env["GITHUB_WORKFLOW_FILE"] = str(workflow_out)

    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert "gh secret set GCP_SA_KEY" in output
    assert "gh secret set GCP_PROJECT_ID" in output
    assert "gh secret set GCP_REGION" in output
    assert str(workflow_out) in output
    assert "Dry run complete" in output
    assert not workflow_out.exists()


def test_setup_workflow_only_writes_valid_yaml(tmp_path: Path) -> None:
    """--workflow-only scaffolds a parseable workflow file."""
    workflow_out = tmp_path / "workflows" / "main.yml"
    env = os.environ.copy()
    env["GITHUB_WORKFLOW_FILE"] = str(workflow_out)

    subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--workflow-only", "--force"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert workflow_out.is_file()
    content = workflow_out.read_text(encoding="utf-8")
    assert "gcloud run deploy" in content

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return

    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    assert "deploy" in data.get("jobs", {})


def test_setup_secrets_only_with_mock_gh(tmp_path: Path) -> None:
    """--secrets-only sets deploy secrets via gh without writing workflow."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_gh(fake_bin)

    sa_key = tmp_path / "sa-key.json"
    sa_key.write_text('{"type":"service_account"}', encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GCP_PROJECT_ID"] = "mock-gcp-project"
    env["GCP_REGION"] = "us-central1"
    env["GCP_SA_KEY_FILE"] = str(sa_key)

    subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--secrets-only"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_deploy_doc_documents_github_cicd_setup() -> None:
    """DEPLOY.md documents setup_github_cicd.sh prerequisites."""
    content = DEPLOY_DOC.read_text(encoding="utf-8")

    assert "setup_github_cicd.sh" in content
    assert "gh auth login" in content
    assert "GCP_SA_KEY" in content
    assert "main.yml" in content
