"""E2E fixtures: uvicorn server on :8080 and pytest-playwright base_url."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
from playwright.sync_api import Page
from tests.e2e.e2e_app import E2E_PORT
from tests.e2e.helpers import reset_e2e_session

E2E_BASE_URL = f"http://localhost:{E2E_PORT}"


@pytest.fixture(scope="session")
def base_url() -> str:
    """pytest-playwright navigates relative to this URL."""
    return E2E_BASE_URL


def _e2e_health_ok() -> bool:
    try:
        response = httpx.get(
            f"{E2E_BASE_URL}/health",
            timeout=5.0,
            follow_redirects=True,
        )
        return response.status_code in {200, 503}
    except httpx.HTTPError:
        return False


def _free_e2e_port() -> None:
    """Stop any process bound to the E2E port so pytest always runs fresh mock code."""
    subprocess.run(
        ["make", "stop-all"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        check=False,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def _isolated_e2e_session(page: Page, base_url: str) -> None:
    """Fresh fakeredis checkpoint + session cookie before each browser E2E test."""
    reset_e2e_session(page, base_url)


@pytest.fixture(scope="session", autouse=True)
def e2e_server() -> Iterator[None]:
    """Start the mocked E2E uvicorn server before Playwright tests."""
    _free_e2e_port()
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
        if _e2e_health_ok():
            break
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
