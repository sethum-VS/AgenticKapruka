"""Structural checks for the production multi-stage Dockerfile."""

from __future__ import annotations

from pathlib import Path

DOCKERFILE = Path("Dockerfile")


def test_dockerfile_exists() -> None:
    assert DOCKERFILE.is_file()


def test_dockerfile_multi_stage_production_contract() -> None:
    content = DOCKERFILE.read_text(encoding="utf-8")

    assert "AS builder" in content
    assert "AS runtime" in content
    assert "python:3.12-slim" in content
    assert "pip install" in content
    assert "tailwindcss" in content
    assert "static/css/app.css" in content
    assert "USER app" in content
    assert "PORT=8080" in content
    assert "EXPOSE 8080" in content
    assert "gunicorn.conf.py" in content
    assert 'CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]' in content
