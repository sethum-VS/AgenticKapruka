"""E2E fixtures: uvicorn server on :8080 and pytest-playwright base_url."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
from tests.e2e.e2e_app import E2E_PORT

E2E_BASE_URL = f"http://localhost:{E2E_PORT}"


@pytest.fixture(scope="session")
def base_url() -> str:
    """pytest-playwright navigates relative to this URL."""
    return E2E_BASE_URL


@pytest.fixture(scope="session", autouse=True)
def e2e_server() -> Iterator[None]:
    """Start the mocked E2E uvicorn server before Playwright tests."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.e2e.e2e_server"],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.wait(timeout=5)
            pytest.fail(f"E2E server exited early: {stderr}")
        try:
            response = httpx.get(f"{E2E_BASE_URL}/health", timeout=2.0)
            if response.status_code in {200, 503}:
                break
        except httpx.HTTPError:
            time.sleep(0.25)
    else:
        proc.kill()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.wait(timeout=5)
        pytest.fail(f"E2E server did not become reachable within 45s: {stderr}")

    yield

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
