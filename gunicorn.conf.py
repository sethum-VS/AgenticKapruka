"""Gunicorn production settings for Cloud Run (SSE chat streams).

Invoked from the Docker image CMD::

    gunicorn -c gunicorn.conf.py app.main:app

Cloud Run sets ``PORT``; workers scale with container vCPU count.
"""

from __future__ import annotations

import multiprocessing
import os

_port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{_port}"

# Standard Gunicorn heuristic; Cloud Run scales horizontally via instance count.
workers = multiprocessing.cpu_count() * 2 + 1

worker_class = "uvicorn.workers.UvicornWorker"

# Long-lived SSE streams from POST /chat/stream.
timeout = 120

# Allow in-flight requests to finish during Cloud Run instance drain.
graceful_timeout = 30

# HTTP keep-alive for GCP load balancer connection reuse.
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
