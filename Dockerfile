# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.3-python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ann --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ann

FROM python:3.12.11-slim-bookworm AS runtime
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    RECOMMENDER__ENVIRONMENT=production \
    RECOMMENDER__PATHS__DATA_DIR=/app/data \
    RECOMMENDER__PATHS__ARTIFACT_DIR=/app/artifacts \
    RECOMMENDER__PATHS__REPORT_DIR=/app/reports
RUN groupadd --system --gid 10001 recommender && useradd --system --uid 10001 --gid recommender --home /nonexistent recommender
WORKDIR /app
COPY --from=builder --chown=recommender:recommender /app/.venv /app/.venv
COPY --chown=recommender:recommender configs /app/configs
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
ENTRYPOINT ["recommender"]
CMD ["serve", "--config", "/app/configs/production.yaml"]
