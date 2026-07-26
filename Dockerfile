# Multi-stage production image for Cloud Run (CPU / NetworkX analytics path).
#
# Build:  docker build -t agentic-kapruka .
# Run:    docker run --rm -p 8080:8080 --env-file .env agentic-kapruka
#
# GPU cuGraph experiments use Dockerfile.cuda (dev-only, not Cloud Run).

# ── Builder: Python deps + Tailwind CSS compile ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml README.md ./
COPY app/ app/
COPY lib/ lib/
COPY graphs/ graphs/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

# Bake cross-encoder weights into the image so Heroku boot does not download from HuggingFace.
ENV HF_HOME=/opt/huggingface
RUN mkdir -p /opt/huggingface \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Compile Tailwind CSS (standalone CLI — no Node.js).
COPY static/css/input.css static/css/input.css
COPY static/img/ static/img/
COPY static/js/ static/js/
COPY tailwind.config.js tailwind.config.js
COPY templates/ templates/
COPY scripts/install-tailwind.sh scripts/install-tailwind.sh

RUN chmod +x scripts/install-tailwind.sh \
    && ./scripts/install-tailwind.sh \
    && bin/tailwindcss -i static/css/input.css -o static/css/app.css --minify \
    && test -s static/css/app.css

# app/* resolves templates/ and static/ via Path(__file__).parent.parent (site-packages).
RUN SITE_PACKAGES="$(python -c "import site; print(site.getsitepackages()[0])")" \
    && cp -r templates static "${SITE_PACKAGES}/"

# ── Runtime: slim image, non-root, gunicorn + uvicorn worker ────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/huggingface /opt/huggingface

COPY --chown=app:app gunicorn.conf.py .

USER app


# Production entrypoint — see gunicorn.conf.py for workers, timeouts, and bind.
CMD sh -c 'if [ -n "$GCP_CREDENTIALS" ]; then echo "$GCP_CREDENTIALS" > /tmp/gcp.json; export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp.json; fi; exec gunicorn -c gunicorn.conf.py app.main:app'
