"""Tests for gunicorn.conf.py production entrypoint."""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import httpx
import pytest

GUNICORN_CONF = Path("gunicorn.conf.py")

_VALID_ENV: dict[str, str] = {
    "REDIS_URL": "redis://localhost:6379/0",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "test-password",
    "ZEP_API_KEY": "zep-test-key",
    "GCP_PROJECT_ID": "test-project",
    "GCP_LOCATION": "us-central1",
    "KAPRUKA_MCP_URL": "https://mcp.kapruka.com/mcp",
    "SESSION_SECRET": "x" * 32,
    "APP_ENV": "development",
}


def _load_gunicorn_conf(*, port: str | None = None) -> ModuleType:
    """Load gunicorn.conf.py as a module (filename contains a dot)."""
    env = os.environ.copy()
    if port is not None:
        env["PORT"] = port
    spec = importlib.util.spec_from_file_location("gunicorn_conf", GUNICORN_CONF)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _patched_environ(env):
        spec.loader.exec_module(module)
    return module


class _patched_environ:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values
        self._original: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self._values.items():
            self._original[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, *args: object) -> None:
        for key, original in self._original.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def test_gunicorn_conf_exists() -> None:
    assert GUNICORN_CONF.is_file()


def test_gunicorn_workers_timeout_and_keepalive() -> None:
    conf = _load_gunicorn_conf(port="8080")

    assert conf.bind == "0.0.0.0:8080"
    assert conf.workers == multiprocessing.cpu_count() * 2 + 1
    assert conf.worker_class == "uvicorn.workers.UvicornWorker"
    assert conf.timeout == 120
    assert conf.graceful_timeout == 30
    assert conf.keepalive == 5


def test_gunicorn_bind_honors_port_env() -> None:
    conf = _load_gunicorn_conf(port="9090")
    assert conf.bind == "0.0.0.0:9090"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_gunicorn_starts_and_serves_health() -> None:
    """Gunicorn process serves GET /health (JSON readiness probe)."""
    port = _free_port()
    env = {**os.environ, **_VALID_ENV, "PORT": str(port)}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "-c",
            str(GUNICORN_CONF),
            "--workers",
            "1",
            f"--bind=127.0.0.1:{port}",
            "app.main:app",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        response: httpx.Response | None = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                pytest.fail(f"gunicorn exited early: {stderr}")
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                break
            except httpx.HTTPError:
                time.sleep(0.25)

        assert response is not None, "gunicorn did not become reachable within 30s"
        assert response.status_code in {200, 503}
        body = response.json()
        assert body["status"] in {"healthy", "degraded"}
        assert set(body["services"]) == {"redis", "neo4j", "zep", "mcp"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
