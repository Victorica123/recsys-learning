# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RECSYS_FEEDBACK_DB=/app/runtime/feedback/events.sqlite3 \
    RECSYS_METRICS_DIR=/app/runtime/metrics/container

WORKDIR /app

RUN groupadd --system recsys && useradd --system --gid recsys recsys
COPY pyproject.toml README.md ./
COPY src ./src
COPY serve.py ./
COPY web ./web
COPY artifacts/release_manifest.json ./artifacts/release_manifest.json

# CPU-only inference image. Data and checkpoints are mounted at runtime and are
# deliberately not baked into an image layer.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && pip install "." \
    && mkdir -p /app/data /app/checkpoints /app/runtime \
    && chown -R recsys:recsys /app

USER recsys
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" || exit 1

CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000", "--preload", "--workers", "2", "--max-in-flight", "8", "--rate-limit", "80", "--rate-burst", "8"]
