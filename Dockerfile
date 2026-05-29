# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
# uv installs project + runtime deps into a self-contained venv at /opt/venv.
FROM python:3.12-slim AS builder

# Pin uv to 0.11.16 for reproducibility (PA7 of PLAN v1.0.1, bumped in v1.0.4).
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Project metadata + sources. pyproject.toml first so dependency-only layers
# stay cacheable when only source changes.
COPY pyproject.toml ./
COPY src ./src

# Create venv and install the project (resolves runtime deps from pyproject).
RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python .


# ---------- Stage 2: runtime ----------
# HEALTHCHECK uses Python stdlib (urllib) — python:3.12-slim ships no curl/wget,
# and adding apt packages just for the probe would inflate the image.
FROM python:3.12-slim AS runtime

# Non-root user (uid 1000) for runtime — defense in depth.
RUN useradd --create-home --uid 1000 --shell /bin/bash app

# Bring over the prebuilt virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application sources and operational scripts (ingest, run_questions).
COPY --chown=app:app src /app/src
COPY --chown=app:app scripts /app/scripts

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Pre-create the volume mount points (/app/pdfs for ingested PDFs,
# /data for the SQLite database) owned by app. Docker copies the
# directory's ownership/perms from the image when a named volume is
# empty on first mount, so the volumes inherit app:app and the
# non-root runtime can write to them.
RUN mkdir -p /app/pdfs /data && chown app:app /app/pdfs /data

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "papers_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
