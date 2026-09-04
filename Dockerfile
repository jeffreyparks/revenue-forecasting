# syntax=docker/dockerfile:1.7
# Multi-stage build for the forecasting API.
# Final image runs `uvicorn api.main:app` with the model + variable data baked in.

FROM public.ecr.aws/docker/library/python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime libs needed by numpy / scipy / scikit-learn wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- builder ----------
FROM base AS builder
WORKDIR /build
COPY api/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- runtime ----------
FROM base AS runtime

# Non-root user
RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /install /usr/local

WORKDIR /app

# Application code. `api` is the FastAPI package; `functions` is referenced
# by the pickled sklearn pipelines (statsmodels wrappers in functions/lr_models.py).
COPY api/ /app/api/
COPY functions/ /app/functions/
# Override the stale api/data/variables copy with the canonical source
COPY data/train/promo_dummies.csv /app/api/data/variables/promo_dummies.csv

USER app

ENV PYTHONPATH=/app \
    PORT=8000

EXPOSE 8000

# `api/main.py` resolves model and feature paths relative to CWD
# (`data/models`, `data/variables`), so run from inside the api/ folder.
WORKDIR /app/api

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/health',timeout=3).status==200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
