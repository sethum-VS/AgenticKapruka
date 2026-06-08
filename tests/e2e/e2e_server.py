"""Run the E2E test server on localhost:8080 (invoked by tests/e2e/conftest.py)."""

from __future__ import annotations

import uvicorn
from tests.e2e.e2e_app import E2E_PORT, create_e2e_app

if __name__ == "__main__":
    app = create_e2e_app()
    uvicorn.run(app, host="127.0.0.1", port=E2E_PORT, log_level="warning")
