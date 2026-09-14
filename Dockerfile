# Pin the verified Python image; normal source rebuilds reuse the same base.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    RECSYS_FEEDBACK_DB=/app/runtime/feedback/events.sqlite3 \
    RECSYS_METRICS_DIR=/app/runtime/metrics/container

WORKDIR /app

RUN groupadd --system recsys && useradd --system --gid recsys recsys
COPY pyproject.toml ./

# CPU-only inference image. Data and checkpoints are mounted at runtime and are
# deliberately not baked into an image layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import pathlib,tomllib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']; pathlib.Path('/tmp/runtime-requirements.txt').write_text('\n'.join(x for x in d if not x.startswith('torch==')))" \
    && pip install -r /tmp/runtime-requirements.txt \
    && mkdir -p /app/data /app/checkpoints /app/runtime \
    && chown -R recsys:recsys /app

# Keep dependency installation cached when only application code changes.
COPY --chown=recsys:recsys src ./src
COPY --chown=recsys:recsys serve.py ./
COPY --chown=recsys:recsys web ./web
COPY --chown=recsys:recsys artifacts/release_manifest.json ./artifacts/release_manifest.json

USER recsys
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" || exit 1

CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000", "--preload", "--workers", "2", "--max-in-flight", "8", "--rate-limit", "80", "--rate-burst", "8"]
