"""Tests for scripts/bootstrap_env.sh local .env bootstrap."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SCRIPT = REPO_ROOT / "scripts" / "bootstrap_env.sh"
GITIGNORE = REPO_ROOT / ".gitignore"


def test_gitignore_ignores_env_but_keeps_example() -> None:
    """`.env` and `.env.*` are gitignored with `!.env.example` exception."""
    content = GITIGNORE.read_text(encoding="utf-8")
    lines = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert ".env" in lines
    assert ".env.*" in lines
    assert "!.env.example" in lines


def test_bootstrap_env_script_contract() -> None:
    """Bootstrap script uses gcloud, python secrets, and required env defaults."""
    assert BOOTSTRAP_SCRIPT.is_file()
    mode = BOOTSTRAP_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "bootstrap_env.sh should be executable"

    content = BOOTSTRAP_SCRIPT.read_text(encoding="utf-8")

    assert "gcloud config get-value project" in content
    assert "gcloud config get-value compute/region" in content
    assert "generativelanguage.googleapis.com" in content
    assert "api-keys create" in content
    assert "api-keys get-key-string" in content
    assert "secrets.token_urlsafe" in content
    assert "REDIS_URL=redis://localhost:6379/0" in content
    assert "KAPRUKA_MCP_URL=" in content
    assert "NEO4J_URI=bolt+s://xxxxxxxx.databases.neo4j.io" in content
    assert "ZEP_API_KEY=your-zep-api-key" in content


def _write_fake_gcloud(fake_bin: Path) -> None:
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "config" && "${2:-}" == "get-value" ]]; then
  case "${3:-}" in
    project) echo "mock-gcp-project" ;;
    compute/region) echo "us-central1" ;;
    *) echo "(unset)" ;;
  esac
  exit 0
fi

if [[ "${1:-}" == "services" && "${2:-}" == "api-keys" ]]; then
  case "${3:-}" in
    list)
      # No existing key — force create path.
      exit 0
      ;;
    create)
      echo "projects/mock-gcp-project/locations/global/keys/mock-key-1"
      exit 0
      ;;
    get-key-string)
      echo "mock-google-api-key-from-gcloud"
      exit 0
      ;;
  esac
fi

echo "fake gcloud: unexpected invocation: $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_gcloud.chmod(fake_gcloud.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_bootstrap_env_generates_settings_compatible_env(tmp_path: Path) -> None:
    """Mocked bootstrap writes .env that pydantic Settings accepts."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_gcloud(fake_bin)

    env_file = tmp_path / ".env"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["BOOTSTRAP_ENV_FILE"] = str(env_file)

    subprocess.run(
        ["bash", str(BOOTSTRAP_SCRIPT), "--force"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert env_file.is_file()
    content = env_file.read_text(encoding="utf-8")
    assert "REDIS_URL=redis://localhost:6379/0" in content
    assert "GOOGLE_API_KEY=mock-google-api-key-from-gcloud" in content
    assert "GCP_PROJECT_ID=mock-gcp-project" in content
    assert "GCP_LOCATION=us-central1" in content
    assert "KAPRUKA_MCP_URL=https://mcp.kapruka.com/mcp" in content
    assert "SESSION_SECRET=" in content

    settings = Settings(_env_file=env_file)
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.google_api_key == "mock-google-api-key-from-gcloud"
    assert settings.gcp_project_id == "mock-gcp-project"
    assert settings.gcp_location == "us-central1"
    assert len(settings.session_secret) >= 32


def test_bootstrap_env_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """Existing .env blocks bootstrap unless --force is passed."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_gcloud(fake_bin)

    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING=1\n", encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["BOOTSTRAP_ENV_FILE"] = str(env_file)

    result = subprocess.run(
        ["bash", str(BOOTSTRAP_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert env_file.read_text(encoding="utf-8") == "EXISTING=1\n"
