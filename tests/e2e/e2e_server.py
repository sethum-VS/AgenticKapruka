"""Run the E2E test server on localhost:8080 (invoked by tests/e2e/conftest.py)."""

from __future__ import annotations

import logging
import os

import uvicorn
from tests.e2e.e2e_app import E2E_PORT, create_e2e_app

if __name__ == "__main__":
    # Optional file logging for debugging routing decisions during Playwright runs.
    log_file = os.environ.get("E2E_LOG_FILE")
    log_level = "info" if log_file else "warning"
    if log_file:
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    app = create_e2e_app()
    uvicorn.run(app, host="127.0.0.1", port=E2E_PORT, log_level=log_level)
